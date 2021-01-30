
from tqdm import tqdm

def train(model,device,train_loader,criterion,optimizer,epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0

  for batch_idx,(data, target) in enumerate(pbar):

    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()

    y_pred = model(data)

    loss = criterion(y_pred,target)

    loss.backward()
    optimizer.step()

    pred= y_pred.argmax(dim=1,keepdim=True)
    correct+= pred.eq(target.view_as(pred)).sum().item()
    processed+= len(data)

    pbar_str = f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}'
    pbar.set_description(desc= pbar_str)