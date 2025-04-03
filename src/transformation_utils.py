import torch
import torch.nn as nn



def scaler_layer(scaler_list):
    if(scaler_list is not None):
        lambda_values = torch.tensor(scaler_list.lambdas_)
        mean_ = torch.tensor(scaler_list._scaler.mean_)
        std_ = torch.tensor(scaler_list._scaler.scale_)
        
        scaling_layer = BoxCoxTransform(lambda_values, mean=mean_, std=std_)
        return scaling_layer
    else:
        return None


class StaticTransformationLayer(nn.Module):
    def __init__(self, transformation):
        super(StaticTransformationLayer, self).__init__()
        self.transformation = transformation

    def forward(self, x):
        # Apply the transformation
        x_transformed = self.transformation(x)
        return x_transformed

####### transformation: box cox ###
class BoxCoxTransform(torch.nn.Module):
    def __init__(self, lambda_values, mean=None, std=None):
        super(BoxCoxTransform, self).__init__()
        self.lambda_values = lambda_values
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # self.mean = mean
        # self.std =  std
        # self.mean = torch.tensor(mean) #
        # self.std =  torch.tensor(std)

        # ✅ Ensure gradients flow
        self.mean = mean.clone().detach().requires_grad_(True).to(self.device) if mean is not None else None
        self.std = std.clone().detach().requires_grad_(True).to(self.device) if std is not None else None


    def forward(self, x):
        # Applying Box-Cox transformation
        transformed = [] 
        
        if self.mean is not None and self.std is not None:
            self.mean = self.mean.clone().detach().to(x.device) # mean #torch.tensor(self.mean)
            self.std = self.std.clone().detach().to(x.device) #std
            
        for i, lambda_value in enumerate(self.lambda_values):
            lambda_value = lambda_value.clone().to(x.device) #torch.tensor(lambda_value).to(x.device)
            if lambda_value.eq(0): 
                # Log transformation for lambda = 0
                transformed.append(torch.log(x[:, i:i+1]))  # Assuming x is 2D tensor
            else:
                # Add epsilon to lambda value to improve numerical stability
                epsilon = 1e-6  # You can adjust this value as needed
                lambda_value += epsilon
                
                # Box-Cox transformation for lambda != 0
                transformed.append(((x[:, i:i+1] ** lambda_value) - 1) / lambda_value)

        if self.mean is not None and self.std is not None:
            return (torch.cat(transformed, dim=1) - self.mean) / (self.std + 1e-6)  # Adding a small epsilon for numerical stability
        else:
            return torch.cat(transformed, dim=1).to(x.device)
    
    def inverse(self, y):
        # Applying inverse Box-Cox transformation
        original = []
        
        if self.mean is not None and self.std is not None:
            self.mean = self.mean.clone().detach().to(y.device) # mean #torch.tensor(self.mean) #.requires_grad_(True)
            self.std = self.std.clone().detach().to(y.device) #std
        
        for i, lambda_value in enumerate(self.lambda_values):
            lambda_value = lambda_value.clone().to(y.device)
            
            if lambda_value.eq(0): 
                if self.mean is not None and self.std is not None:
                    ori = y[:, i:i+1]* (torch.tensor(self.std[i]) + 1e-6) + torch.tensor(self.mean[i])
                    original.append(torch.exp(ori))
                else:
                    original.append(torch.exp(y[:, i:i+1]))  # Assuming y is 2D tensor
            else:
                # Add epsilon to lambda value to improve numerical stability
                epsilon = 1e-6  # You can adjust this value as needed
                lambda_value += epsilon
                
                # Inverse Box-Cox transformation for lambda != 0
                if self.mean is not None and self.std is not None:
                    ori = y[:, i:i+1]* (self.std[i] + 1e-6) + self.mean[i]
                    original.append(((lambda_value * ori) + 1) ** (1 / lambda_value))
                else:
                    original.append(((lambda_value * y[:, i:i+1]) + 1) ** (1 / lambda_value))
        return torch.cat(original, dim=1).to(y.device)