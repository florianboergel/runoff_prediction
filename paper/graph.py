import graphviz

def draw_balt_net():
    dot = graphviz.Digraph(format='png')

    # Add nodes
    dot.node('Input', 'Input Layer')
    dot.node('Encoder', 'ConvLSTM Encoder')
    dot.node('Decoder', 'ConvLSTM Decoder')
    dot.node('Attention', 'Spatial Attention')
    dot.node('Flatten', 'Flatten Output')
    dot.node('FC1', 'Fully Connected Layer 1')
    dot.node('ReLU1', 'ReLU Activation 1')
    dot.node('FC2', 'Fully Connected Layer 2')
    dot.node('ReLU2', 'ReLU Activation 2')
    dot.node('FC3', 'Fully Connected Layer 3')
    dot.node('Output', 'Output Layer')

    # Add edges
    dot.edge('Input', 'Encoder')
    dot.edge('Encoder', 'Decoder')
    dot.edge('Decoder', 'Attention')
    dot.edge('Attention', 'Flatten')
    dot.edge('Flatten', 'FC1')
    dot.edge('FC1', 'ReLU1')
    dot.edge('ReLU1', 'FC2')
    dot.edge('FC2', 'ReLU2')
    dot.edge('ReLU2', 'FC3')
    dot.edge('FC3', 'Output')

    return dot

# Draw the network
balt_net_graph = draw_balt_net()
balt_net_graph.render('BaltNet_Structure')

# The network structure is saved as 'BaltNet_Structure.png' in the current directory

