# MelGAN
TF implementation of MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis, Kumar et al  
# Example Run
The notebook `melgan/MELGAN.ipynb` contains a collab example using a Nvidia Tesla T4 GPU and the LJSpeech dataset.  
After training, the Generator can be used to generate speech from a spectogram. Current implementation is still a **work in progress** becase generated speech contains noise, and when compared to Griffin-Lim reconstruction the quality is lower. Related to this is that during training the generator loss decreases but discriminator loss remains unchanged.  
Despite the above issues, the voice of generated speech matches the ground truth file and words can be totally understood.  


