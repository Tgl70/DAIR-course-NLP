import tensorflow as tf
import time


def pgd_attack_batch(model, x_batch, y_batch, pgd_steps, eps, gamma):
    """
    Performs PGD attack on a batch of inputs.

    Args:
    model: The model to attack.
    x_batch: Batch of input data.
    y_batch: Batch of true labels.
    pgd_steps: Number of PGD steps.
    eps: Tuple containing (lower bounds, upper bounds) for each dimension from hyperrectangles.
    gamma: Step size for each PGD iteration.

    Returns:
    perturbed_x: Adversarial examples generated from the input batch.
    """
    x_adv = tf.identity(x_batch)
    
    for _ in range(pgd_steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            predictions = model(x_adv)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions)
        
        # Compute gradient of the loss w.r.t the input
        gradients = tape.gradient(loss, x_adv)
        
        # Apply perturbation based on sign of the gradient
        signed_grad = tf.sign(gradients)
        x_adv = x_adv + gamma * signed_grad
        
        # Clip the adversarial example to ensure it's within the epsilon range
        x_adv = tf.clip_by_value(x_adv, eps[0], eps[1])  # Clip between lower and upper bounds from hyperrectangles
        x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)  # Ensure values stay between 0 and 1

    return x_adv


def train(model, train_dataset, test_dataset, epochs, batch_size, pgd_steps, hyperrectangles, hyperrectangles_labels, alpha=1, beta=0, gamma_multiplier=1, from_logits=False, optimizer=tf.keras.optimizers.Adam()):
    """
    Trains a model using PGD adversarial training with hyperrectangles.
    
    Args:
    model: The model to train.
    train_dataset: The training dataset.
    test_dataset: The test dataset.
    epochs: Number of epochs to train for.
    batch_size: Batch size for training.
    pgd_steps: Number of steps for PGD attack.
    hyperrectangles: List of hyperrectangles for adversarial examples.
    hyperrectangles_labels: Labels associated with the hyperrectangles.
    alpha: Coefficient for standard cross-entropy loss.
    beta: Coefficient for adversarial loss.
    gamma_multiplier: Multiplier for gamma (from hyperrectangle bounds).
    from_logits: Whether to interpret model output as logits.
    optimizer: The optimizer to use for training.
    
    Returns:
    Trained model.
    """
    ce_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
    pgd_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)

    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    pgd_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    train_loss_metric = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=from_logits)
    test_loss_metric = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=from_logits)
    pgd_loss_metric = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=from_logits)

    # Combine hyperrectangles and labels into a TensorFlow dataset
    hyperrectangles = tf.convert_to_tensor(hyperrectangles, dtype=tf.float32)
    hyperrectangles_labels = tf.convert_to_tensor(hyperrectangles_labels, dtype=tf.int64)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                # Forward pass for clean examples
                outputs = model(x_batch_train, training=True)
                ce_loss_value = ce_loss_fn(y_batch_train, outputs)
                total_loss = alpha * ce_loss_value

            # Compute gradients and update weights
            grads = tape.gradient(total_loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        ######################################### PGD ####################################################
        # Doing the PGD training loop only if 'beta' is > 0. Otherwise the training will be standard.
        pgd_dataset = []
        pgd_labels = []
        if beta > 0:
            for i, hyperrectangle in enumerate(hyperrectangles):
                # Extract the min and max bounds for each dimension (eps), transpose the hyperrectangle
                t_hyperrectangle = tf.transpose(hyperrectangle)

                # Calculate gamma (step size) from the hyperrectangle bounds
                gamma = tf.expand_dims((t_hyperrectangle[1] - t_hyperrectangle[0]) / (pgd_steps * gamma_multiplier), axis=0)

                # Generate random points in hyperrectangles
                pgd_point = tf.random.uniform(shape=[1, t_hyperrectangle.shape[-1]],
                                              minval=t_hyperrectangle[0],
                                              maxval=t_hyperrectangle[1])
                pgd_label = tf.expand_dims(hyperrectangles_labels[i], axis=0)

                # Run the PGD attack to generate adversarial examples
                pgd_point = pgd_attack_batch(model, pgd_point, pgd_label, pgd_steps, t_hyperrectangle, gamma)

                if len(pgd_dataset) > 0:
                    pgd_dataset = tf.concat([pgd_dataset, pgd_point], axis=0)
                    pgd_labels = tf.concat([pgd_labels, pgd_label], axis=0)
                else:
                    pgd_dataset = pgd_point
                    pgd_labels = pgd_label
            
            # Convert the pgd generated inputs into tf datasets, shuffle and batch them
            pgd_dataset = tf.data.Dataset.from_tensor_slices((pgd_dataset, pgd_labels))
            pgd_dataset = pgd_dataset.shuffle(buffer_size=1024).batch(batch_size)

            # Iterate over the batches of the pgd dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(pgd_dataset): 
                with tf.GradientTape() as tape:
                    pgd_outputs = model(x_batch_train, training=True)
                    pgd_loss_value = pgd_loss_fn(y_batch_train, pgd_outputs)
                    pgd_loss_value = beta * pgd_loss_value

                grads = tape.gradient(pgd_loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
        ################################################################################################
        
        # Run a training loop at the end of each epoch.
        for x_batch_train, y_batch_train in train_dataset:
            train_outputs = model(x_batch_train, training=False)
            train_acc_metric.update_state(y_batch_train, train_outputs)
            train_loss_metric.update_state(y_batch_train, train_outputs)

        # Run a testing loop at the end of each epoch.
        for x_batch_test, y_batch_test in test_dataset:
            test_outputs = model(x_batch_test, training=False)
            test_acc_metric.update_state(y_batch_test, test_outputs)
            test_loss_metric.update_state(y_batch_test, test_outputs)

        # Run a pgd loop at the end of each epoch.
        for x_batch_test, y_batch_test in pgd_dataset:
            pgd_outputs = model(x_batch_test, training=False)
            pgd_acc_metric.update_state(y_batch_test, pgd_outputs)
            pgd_loss_metric.update_state(y_batch_test, pgd_outputs)

        train_acc = train_acc_metric.result().numpy()
        test_acc = test_acc_metric.result().numpy()
        pgd_acc = pgd_acc_metric.result().numpy()
        train_loss = train_loss_metric.result().numpy()
        test_loss = test_loss_metric.result().numpy()
        pgd_loss = pgd_loss_metric.result().numpy()

        train_acc_metric.reset_state()
        test_acc_metric.reset_state()
        pgd_acc_metric.reset_state()
        train_loss_metric.reset_state()
        test_loss_metric.reset_state()
        pgd_loss_metric.reset_state()

        # train_acc = float(train_acc)
        # test_acc = float(test_acc)
        # pgd_acc = float(pgd_acc)

        # train_loss = float(train_loss)
        # test_loss = float(test_loss)
        # pgd_loss = float(pgd_loss)

        print(f"Train acc: {train_acc:.4f}, Train loss: {train_loss:.4f} --- Test acc: {test_acc:.4f}, Test loss: {test_loss:.4f} --- PGD acc: {pgd_acc:.4f}, PGD loss: {pgd_loss:.4f} --- Time: {(time.time() - start_time):.2f}s")
    return model