import json
import os
from nemo.collections.asr.models import NeuralDiarizer
from omegaconf import OmegaConf
import wget

def diarize_audio(input_file):
    # Diarization configuration
    meta = {
        'audio_filepath': input_file,
        'offset': 0, 
        'duration': None, 
        'label': 'infer', 
        'text': '-',
        'num_speakers': None, 
        'rttm_filepath': None,
        'uem_filepath': None 
    }

    # Write manifest
    with open('input_manifest.json', 'w') as fp:
        json.dump(meta, fp)
        fp.write('\n')

    output_dir = os.path.join('output')
    os.makedirs(output_dir, exist_ok=True)

    # Load model config
    model_config = 'diar_infer_telephonic.yaml'
    if not (model_config):
        config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_general.yaml"
        model_config = wget.download(config_url)# Update the path to the MSDD model configuration
    config = OmegaConf.load(model_config)
    
    
    config.diarizer.msdd_model.model_path = 'diar_msdd_telephonic' # telephonic speaker diarization model 
    config.diarizer.msdd_model.parameters.sigmoid_threshold = [0.7, 1.0] # Evaluate with T=0.7 and T=1.0
    
    # Initialize diarizer
    msdd_model = NeuralDiarizer(cfg=config)

    # Diarize audio
    diarization_result = msdd_model.diarize()

    return diarization_result

if __name__ == "__main__":
    input_file = 'obama_zach(sample).wav'  # mono .wav
    result = diarize_audio(input_file)
