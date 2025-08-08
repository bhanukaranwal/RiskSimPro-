import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px

def plot_loss_distribution(losses, bins=50, filename=None):
    plt.figure(figsize=(10, 6))
    plt.hist(losses, bins=bins, color='blue', alpha=0.7)
    plt.title('Loss Distribution')
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def interactive_dashboard(data):
    fig = px.histogram(data, nbins=50, title='Interactive Loss Distribution')
    fig.show()
