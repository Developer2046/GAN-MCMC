Preserving Temporal Dynamics in Synthetic Multivariate Time Series Using Generative Neural Networks and Monte Carlo Markov Chain


Overview

This project aims to generate multivariate time series data by leveraging two models:

1. **Generative Adversarial Networks (GANs)** - Used to generate the time source.
2. **Markov Chain Monte Carlo (MCMC)** - Utilized to further process the GAN-generated data and generate multivariate time series data.

Getting Started

1. **Generate the DataSource along with TimeStamp:**
   - Ensure you have the necessary libraries installed (e.g., TensorFlow/PyTorch, NumPy, etc.).
   - Run the `GAN_Generate_DataSource.ipynb` to generate the time source using the GAN model.
   
2. **Processing the DataSource from the GAN model and Sampling the DataSource the Second Time to Generate Time Series:**
   - Once the time source is generated, use the `MCMC_Generated_TimeSeries.ipynb` to run the MCMC algorithm and generate the multivariate time series data.

Troubleshooting

If you encounter any issues with the code or have questions, feel free to contact me at **cilin2046@gmail.com**.

