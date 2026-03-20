import os
import sys
import shutil
import torch
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from codon.utils.dataset.conflux.base import ConfluxDataset
from codon.utils.dataset.conflux.reader import ConfluxReader


def run_test() -> None:
    '''
    Executes the end-to-end test for the Conflux dataset pipeline.

    This function sets up a dataset schema, generates dummy data, writes
    the data to multiple shards, and subsequently reads and verifies it
    using the ConfluxReader.
    '''
    test_dir = './test_dataset'
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    print('Initializing ConfluxDataset...')
    dataset = ConfluxDataset(test_dir)

    dataset.set_info(
        name='test_conflux',
        version='1.0.0',
        description='A simple test dataset.',
        compression_mode='tar.gz',
        max_samples_per_shard=2
    )

    dataset.add_schema(
        key_name='image',
        modality='image',
        extension='npy',
        dtype='float32',
        shape=[3, 64, 64]
    )

    dataset.add_schema(
        key_name='text',
        modality='text',
        extension='txt',
        dtype='str'
    )

    dataset.add_schema(
        key_name='label',
        modality='label',
        extension='pt',
        dtype='int64',
        shape=[1]
    )

    dataset.add_task(
        task_name='classification',
        inputs=['image', 'text'],
        targets=['label'],
        description='Image classification with text metadata.'
    )

    print('Writing data to shards...')
    num_samples = 5
    with dataset.open() as writer:
        for i in range(num_samples):
            dummy_image = np.random.rand(3, 64, 64).astype(np.float32)
            dummy_text = f'Sample text {i}'
            dummy_label = torch.tensor([i], dtype=torch.int64)

            sample_data = {
                'image': dummy_image,
                'text': dummy_text,
                'label': dummy_label
            }
            
            writer.write(sample_data)

    print('Reading data from shards...')
    loaded_dataset = ConfluxDataset(test_dir)
    reader = ConfluxReader(loaded_dataset, task_name='classification')
    
    dataloader = reader.loader(batch_size=1)

    sample_count = 0
    for batch in dataloader:
        inputs, targets = batch
        
        image_batch = inputs['image']
        text_batch = inputs['text']
        label_batch = targets['label']
        
        print(f'Batch {sample_count}:')
        print(f"  Image shape: {image_batch.shape}")
        print(f"  Text: {text_batch}")
        print(f"  Label shape: {label_batch.shape}")
        
        assert image_batch.shape == (1, 3, 64, 64), 'Image shape mismatch.'
        assert label_batch.shape == (1, 1), 'Label shape mismatch.'
        
        sample_count += 1
        
    assert sample_count == num_samples, 'Number of read samples does not match written samples.'
    print('Test passed successfully.')


if __name__ == '__main__':
    run_test()
