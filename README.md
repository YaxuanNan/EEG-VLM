1.Reference source

     EEG-VLM project:https://github.com/Elijah123463/EEG-VLM
     
     Large Language Model Llava-v15-13b：https://llava-vl.github.io/
     
     Visual language models clip vit large patch 14-336 and ResNet18 (framework needs to be written by oneself):
     
        Clip vit large patch 14-336 weight: https://huggclip-vit-large-patch14-336
     
        ResNet18 reference：Thesis，"Deep Residual Learning for Image Recognition"
     
     Classification of Sleep Stage Dataset:Sleep-EDF Database Expanded v1.0.0
   

2.Related instructions  

  The main program path for filtering the sleep stage dataset is:  
  
    /EGG/EG_SLEEP_datasets/process_stleep.data/sleep-efd-converter  
  
  The main program path for feature extraction and partitioning of the dataset into training and validation sets is as follows:  
  
    EEG/EEG_SLEEP_datasets/process_visual_resnet18/src  
    
  The path where the filtered data, training set, validation set feature files, JSON files required for training evaluation, and other information are generated:  
  
    /EEG/EEG_SLEEP_datasets/process_sleep_data/sleep_event  
    
  EEG-VLM Engineering Catalog: /EEG/EEG-VLM-main    


3.Install  

    conda create -n eeg_vlm_backup python=3.10 -y  
    
    conda activate eeg_vlm_backup  
    
    pip install --upgrade pip  
    
    pip install -e .  
    
    python -c "import torch; print('PyTorch is fixed! Version:', torch.__version__)"  
    
    pip install -e ".[train]"  
    
    pip install flash-attn --no-build-isolation  
    
    #if you see some import errors when you upgrade,  
    
    #please try running the command below (without #)  
    
    #MAX_JOBS=2 pip install flash-attn==1.0.3.post0 --no-build-isolation --no-cache-dir  or  
    
    #pip install flash-attn==1.0.3.post0 --no-build-isolation --no-cache-dir  
  
  
