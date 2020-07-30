# Conv-GCN
Code for Multi-graph convolutional network for short-term passenger flow forecasting in urban rail transit
## [Multi-graph convolutional network for short-term passenger flow forecasting in urban rail transit](10.1049/iet-its.2019.0873)

<img src="https://github.com/JinleiZhangBJTU/Conv-GCN/blob/master/pictures/model.png" width = "722" height = "813" alt="model structure" 
align=left>

## Data

The dimension of inflow data is n*time steps, where n represents number of stations and time steps denote time steps in 25 weekdays.

The structure of outflow data is the same with inflow data.

## Requirement

Keras == 2.2.4  
tensorflow-gpu == 1.10.0  
numpy == 1.14.5  
scipy == 1.3.3  
scikit-learn == 0.20.2  
protobuf == 3.6.0  

## Implement

Just download this repository and using PyCharm to open it. Then run Conv-GCN.py.

## Result

![Model comparison](https://github.com/JinleiZhangBJTU/Conv-GCN/blob/master/pictures/result.png)

## Reference

Zhang, Jinlei; Chen, Feng; Guo, Yinan; Li, Xiaohong: '[Multi-graph convolutional network for short-term passenger flow forecasting in urban rail transit](10.1049/iet-its.2019.0873)', IET Intelligent Transport Systems, 2020, DOI: 10.1049/iet-its.2019.0873


