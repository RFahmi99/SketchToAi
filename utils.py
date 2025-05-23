import torch


def trainModel(epochs, model, optimizer, scheduler, loss_fn, train_loader, test_loader, device):
    # Resume training if checkpoint exists
    checkpoint_path = 'outputs/best_model.pth'
    try:
        # checkpoint = torch.load(checkpoint_path, weights_only=True)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"Resuming training from epoch {start_epoch}")
    except FileNotFoundError:
        print("No checkpoint found, starting fresh.")
        start_epoch = 0
        best_loss = float('inf')

    for epoch in range(start_epoch, epochs):
    # Training & Testing Loop
        model.train()

        # Initialize epoch loss
        train_loss = 0.0
        for i, (train_sketch, train_original) in enumerate(train_loader):
            # Move data to the device
            train_sketch = train_sketch.to(device)
            train_original = train_original.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            train_output = model(train_sketch)

            # Loss Function & Canculation
            loss = loss_fn(train_output, train_original)

            # Loss backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step 
            optimizer.step()

            # Accumulate loss for the epoch
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Train Loss: {avg_train_loss:.4f}")

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, (test_sketch, test_original) in enumerate(test_loader):
                # Move data to the device
                test_sketch = test_sketch.to(device)
                test_original = test_original.to(device)

                # Forward pass
                test_output = model(test_sketch)

                # Loss Function & Canculation
                loss = loss_fn(test_output, test_original)

                # Accumulate loss for the epoch
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)

        # Scheduler step
        scheduler.step(avg_train_loss)

        # Print average test loss
        print(f"Epoch [{epoch+1}/{epochs}], Average Test Loss: {avg_test_loss:.4f}")

        # Print current learning rate
        for param_group in optimizer.param_groups:
            print(f"Current Learning Rate: {param_group['lr']:.6f}")

        # Save the best model
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss},
                'outputs/best_model.pth'
                )
            
        with open("./outputs/log.txt", "a") as log_file:
            log_file.write(f"Epoch [{epoch+1}/{epochs}], Average Train Loss: {avg_train_loss:.4f}\n")
            log_file.write(f"Epoch [{epoch+1}/{epochs}], Average Test Loss: {avg_test_loss:.4f}\n")