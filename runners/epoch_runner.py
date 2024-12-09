import torch
import tqdm

class EpochRunner:
    def __init__(self, model, device, optimizer=None, loss_func=None, scheduler_handler=None, metrics=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.metrics = metrics if metrics is not None else [] 
        self.scheduler_handler = scheduler_handler
        self.device = device

    def run_epoch(self, mode, epoch_num, dataloader):
        mode = mode.lower()
        if mode == 'train':
            return self.training_epoch(epoch_num, dataloader)
        elif mode in ['validate', 'test']:
            return self.validation_epoch(epoch_num, dataloader, mode)
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'train', 'validate', or 'test'.")

    def to_cpu(self, tensor):
        return tensor.detach().cpu().numpy()

    def compute_metrics(self, gts, preds):
        metric_results = {}
        for metric_func in self.metrics:
            metric_name = metric_func.__name__
            metric_value = metric_func(gts, preds)
            metric_results[metric_name] = metric_value
        return metric_results

    def training_epoch(self, epoch_num, dataloader):
        self.model.train()
        running_loss = 0
        all_preds, all_gts = [], []
        with torch.enable_grad():
            for batch in tqdm.tqdm(dataloader, desc=f'[Epoch {epoch_num+1}   Training]'):
                batch_X, batch_y, *extra = [item.to(self.device) for item in batch]
                
                if extra:
                    batch_pred = self.model(batch_X, *extra)
                else:
                    batch_pred = self.model(batch_X)

                loss = self.loss_func(batch_pred, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.scheduler_handler is not None:
                    schduler = self.scheduler_handler(epoch_num)
                    schduler.step()

                running_loss += loss.item()

                #_, predicted_labels = torch.max(batch_pred, 1)
                all_preds.extend(self.to_cpu(batch_pred))
                all_gts.extend(self.to_cpu(batch_y))

            metrics_dict = self.compute_metrics(all_gts, all_preds)
            epoch_loss = running_loss / len(dataloader)

            return metrics_dict, epoch_loss

    def validation_epoch(self, epoch_num, dataloader, mode):
        self.model.eval()
        running_loss = 0
        all_preds, all_gts, all_feats = [], [], []
        mode_name = "Validating" if mode == 'validate' else "Testing"
        tqdm_desc = f'[Epoch {epoch_num + 1} {mode_name.rjust(10)}]'
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc=tqdm_desc):
                batch_X, batch_y, *extra = [item.to(self.device) for item in batch]

                if extra:
                    batch_pred = self.model(batch_X, *extra)
                else:
                    batch_pred = self.model(batch_X)

                if hasattr(self.model, 'extract_features'):
                    batch_pred, batch_feats = self.model.extract_features(batch_X, *extra)

                loss = self.loss_func(batch_pred, batch_y)

                all_preds.extend(self.to_cpu(batch_pred))
                all_gts.extend(self.to_cpu(batch_y))
                if hasattr(self.model, 'extract_features'):
                    all_feats.extend(self.to_cpu(batch_feats))

                running_loss += loss.item()

            metrics_dict = self.compute_metrics(all_gts, all_preds)
            epoch_loss = running_loss / len(dataloader)

            return metrics_dict, epoch_loss, all_preds, all_gts, all_feats
