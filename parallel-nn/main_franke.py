'''
How to Train Neural Networks in Parallel (From Scratch)
===
Author: Nicholas Geneva (MIT Liscense)
url: https://nicholasgeneva.com/blog/
github: https://github.com/NickGeneva/blog-code
===
'''
import torch
import time
import numpy as np
from distributed import Serial, Mpi, Nccl, Gloo, NcclP
from src_franke.args import Parser
from src_franke.nn import FCNetwork
from src_franke.data import create_training_dataloader, create_testing_dataloader
from src_franke.viz import plot_prediction

if __name__ == '__main__':

    # Parse 
    args = Parser().parse() 
    # Set up distrubted training objects
    if args.comm.lower() == "serial":
        dcomm = Serial()
    elif args.comm.lower() == "mpi":
        dcomm = Mpi()
    elif args.comm.lower() == "nccl":
        dcomm = Nccl()
    elif args.comm.lower() == "ncclp":
        dcomm = NcclP()
    elif args.comm.lower() == "gloo":
        dcomm = Gloo()
    else:
        raise NotImplementedError("Invalid distributed training method.")
    print(f"Neural network will be trained using {args.comm.lower()} on {dcomm.size} processes")

    # Create training and testing dataloaders
    training_loader = create_training_dataloader(
        args.ntrain,
        args.train_batch_size,
        dcomm,
        seed = args.seed
    )
    if dcomm.rank == 0:
        testing_loader = create_testing_dataloader(
            args.ntest,
            args.test_batch_size,
        )

    # Create model and sync params
    model = FCNetwork(2, 1).to(dcomm.device)
    dcomm.copy_model(model)
    print("Number of model parameters: {:d}".format(model.num_parameters()))

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma = 0.98)

    # Training loop
    start_time = time.time()
    valid_losses = []
    loss_fnct = torch.nn.MSELoss()
    for epoch in range(1, args.epochs+1):
        # Loop through minibatches
        training_loss = 0
        for mbi, data in enumerate(training_loader):
            optim.zero_grad()
            input = data['input'].to(dcomm.device)
            target = data['target'].to(dcomm.device)
            # Model forward
            pred = model(input)
            # Loss calculation
            loss = loss_fnct(pred, target)
            training_loss += loss.detach() / len(training_loader)

            loss.backward()
            dcomm.average_gradients(model)

            optim.step()

        scheduler.step()
        print('Epoch {:d}: Training loss : {:.04f}'.format(epoch, training_loss))

        # ==== Validation ====
        if epoch % args.test_freq == 0:
        
            if dcomm.rank == 0:
                with torch.no_grad():
                    valid_loss = 0
                    for mbi, data in enumerate(testing_loader):
                        input = data['input'].to(dcomm.device)
                        target = data['target'].to(dcomm.device)
                        # Model forward
                        pred = model(input)

                        valid_loss += loss_fnct(pred, target).item() / len(testing_loader)

                    valid_losses.append([epoch, valid_loss])
                    print('Epoch {:d}: Validation loss : {:.04f}'.format(epoch, valid_loss))

            dcomm.barrier()

        # ==== Plot Prediction ====
        if epoch == 1 or epoch % args.plot_freq == 0:
            if dcomm.rank == 0:
                plot_prediction(model, epoch, args.pred_dir, device=dcomm.device)
            dcomm.barrier()

        # ==== Checkpoint ====
        if epoch == 1 or epoch+1 % args.ckpt_freq == 0:
            file_name = "model_proc{:d}_{:d}.pt".format(dcomm.rank, epoch)
            torch.save(model.state_dict(), args.ckpt_dir / file_name)

    # Save validation losses and print execution time
    np.save(args.run_dir / "valid", np.array(valid_losses))
    print("Training on {:d} processes complete.".format(dcomm.size))
    print("Training took: {:.01f} sec".format(time.time()-start_time))
    
    comm_error = dcomm.comm_error(model)
    print("Error between device models: {:.04e}".format(comm_error))
