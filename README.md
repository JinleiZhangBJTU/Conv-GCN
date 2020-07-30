# Conv-GCN
Keras code for Multi-graph convolutional network for short-term passenger flow forecasting in urban rail transit
## [Multi-graph convolutional network for short-term passenger flow forecasting in urban rail transit](http://doi.org/10.1049/iet-its.2019.0873)

![model structure](https://github.com/JinleiZhangBJTU/Conv-GCN/blob/master/pictures/model.png)  

## Description  

Short-term passenger flow forecasting is a crucial task for urban rail transit operations. Emerging deep-learning technologies have become effective methods used to overcome this problem. In this study, the authors propose a deep-learning architecture called “**Conv-GCN**” that combines a graph convolutional network (GCN) and a three-dimensional (3D) convolutional neural network (3D CNN). First, they introduce a multi-graph GCN to deal with three inflow and outflow patterns (recent, daily, and weekly) separately. Multi-graph GCN networks can capture spatiotemporal correlations and topological information within the entire network. A 3D CNN is then applied to deeply integrate the inflow and outflow information. High-level spatiotemporal features between different inflow and outflow patterns and between stations that are nearby and far away can be extracted by 3D CNN. Finally, a fully connected layer is used to output results. The Conv-GCN model is evaluated on smart card data of the Beijing subway under the time interval of 10, 15, and 30 min. Results show that this model yields the best performance compared with seven other models. In terms of the root-mean-square errors, the performances under three time intervals have been improved by 9.402, 7.756, and 9.256%, respectively. This study can provide critical insights for subway operators to optimise urban rail transit operations.

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

Zhang, Jinlei; Chen, Feng; Guo, Yinan; Li, Xiaohong: '[Multi-graph convolutional network for short-term passenger flow forecasting in urban rail transit](http://doi.org/10.1049/iet-its.2019.0873)', IET Intelligent Transport Systems, 2020, DOI: 10.1049/iet-its.2019.0873
