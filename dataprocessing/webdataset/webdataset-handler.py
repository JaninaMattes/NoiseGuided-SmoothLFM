import os
import webdataset as wds
import numpy as np
from PIL import Image # Assuming images are PNG or JPEG readable by PIL
import glob # To find directories matching a pattern
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_webdataset(data_dir, output_pattern, max_shard_size=1e9, max_count=100000):
    """
    Creates sharded WebDataset tar files from a structured directory.

    Args:
        data_dir (str): The base directory containing your data splits (e.g., './my_dataset/').
                        Expected structure: data_dir/[split]/Images, data_dir/[split]/Labels, data_dir/[split]/Latents_X.XX
        output_pattern (str): The output pattern for the sharded tar files
                               (e.g., './webdataset_shards/my_dataset-{split}-{0000xx}.tar').
                               Must include '{split}' and '{0000xx}'.
        max_shard_size (int): The maximum size of each shard in bytes.
        max_count (int): The maximum number of samples per shard.
    """
    if '{split}' not in output_pattern or '{0000xx}' not in output_pattern:
         raise ValueError("output_pattern must contain '{split}' and '{0000xx}'")

    # Find all split directories (e.g., 'train', 'validation')
    split_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for split in split_dirs:
        split_dir = os.path.join(data_dir, split)
        logging.info(f"Processing split: {split} from {split_dir}")

        images_dir = os.path.join(split_dir, 'Images')
        labels_dir = os.path.join(split_dir, 'Labels')
        # Find all latent directories matching the pattern Latents_X.XX
        latent_dirs = sorted(glob.glob(os.path.join(split_dir, 'Latents_[0-9].[0-9][0-9]')))

        if not os.path.exists(images_dir):
            logging.warning(f"Skipping {split}: Images directory not found at {images_dir}.")
            continue

        # Determine the output pattern for this specific split
        output_pattern_split = output_pattern.replace('{split}', split)
        logging.info(f"Output pattern for {split}: {output_pattern_split}")

        # Initialize ShardWriter
        # The 'maxcount' and 'maxsize' control when a new tar file (shard) is started.
        shard_writer = wds.ShardWriter(output_pattern_split, maxcount=max_count, maxsize=int(max_shard_size))

        # Get the list of image files to determine sample basenames
        # Assuming image files determine the set of samples
        image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        if not image_files:
            logging.warning(f"No image files found in {images_dir}. Skipping split {split}.")
            shard_writer.close() # Close the writer even if no files were written
            continue

        logging.info(f"Found {len(image_files)} images in {images_dir}.")

        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0] # e.g., '00001'
            sample = {"__key__": base_name} # Unique identifier for the sample

            # Add Image
            image_path = os.path.join(images_dir, img_file)
            try:
                with open(image_path, 'rb') as f:
                    # Use the actual file extension as the key in the tar (lowercase is common)
                    sample[img_file.split('.')[-1].lower()] = f.read()
            except FileNotFoundError:
                logging.warning(f"Image not found for {base_name} at {image_path}, skipping sample.")
                continue # Skip this sample if the main image file is missing
            except Exception as e:
                logging.error(f"Error reading image file {image_path}: {e}")
                continue # Skip sample on read error

            # Add Label if Labels directory exists
            label_path = os.path.join(labels_dir, f'{base_name}.npy')
            if os.path.exists(labels_dir) and os.path.exists(label_path):
                try:
                    label_data = np.load(label_path)
                    # Store as 'label.npy' in the tar
                    # WebDataset can decode .npy automatically using 'decode("npy")' in the reader pipeline
                    sample['label.npy'] = label_data
                except FileNotFoundError:
                     # Should not happen if os.path.exists passed, but good practice
                     logging.warning(f"Label file not found for {base_name} at {label_path}.")
                except Exception as e:
                    logging.error(f"Error loading label for {base_name} from {label_path}: {e}")
                    # Decide how to handle: skip label, skip sample, log error

            # Add Latents from all found latent directories
            for latent_dir in latent_dirs:
                # Extract the timestep part from the directory name (e.g., '0.00', '0.10')
                # Replace '.' with 'p' for safer key names
                timestep_suffix = os.path.basename(latent_dir).replace('Latents_', '').replace('.', 'p')
                latent_file_name = f'{base_name}.npy'
                latent_path = os.path.join(latent_dir, latent_file_name)

                if os.path.exists(latent_path):
                    try:
                        latent_data = np.load(latent_path)
                        # Use a key that includes the timestep, e.g., 'latent_0p00.npy'
                        sample[f'latent_{timestep_suffix}.npy'] = latent_data
                    except FileNotFoundError:
                         logging.warning(f"Latent file not found for {base_name} at {latent_path}.")
                    except Exception as e:
                        logging.error(f"Error loading latent for {base_name} from {latent_path}: {e}")
                        # Decide how to handle error

            # Write the complete sample dictionary to the shard
            if sample: # Only write if at least the image was loaded
                try:
                    shard_writer.write(sample)
                except Exception as e:
                     logging.error(f"Error writing sample {base_name} to shard: {e}")
                     # Decide how to handle write errors

        shard_writer.close() # Close the writer for the current split
        logging.info(f"Finished creating WebDataset shards for split: {split}")

    logging.info("WebDataset creation process finished.")


if __name__ == "__main__":
    # the expected directory structure is:
    #
    # ./my_dataset/train/Images/00001.png
    # ./my_dataset/train/Labels/00001.npy
    # ./my_dataset/train/Latents_0.00/00001.npy
    # ./my_dataset/train/Latents_0.10/00001.npy
    # ... and similarly for validation etc.

    my_dataset_dir = './dataset/processed/imagenet-256/'

    # {split} will be replaced by 'train', 'validation', etc.
    # {0000xx} is the shard number (padded with zeros)
    # Example: './webdataset_shards/my_dataset-train-{000001}.tar'
    output_pattern = './dataset/processed/imagenet-256/wds_shards/imgnet256-{split}-{0000xx}.tar'

    output_dir = os.path.dirname(output_pattern.split('{')[0])
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Ensuring output directory exists: {output_dir}")


    # Run the creation function with the base directory and the template pattern
    try:
        # Adjust max_shard_size or max_count if you want smaller/larger tar files
        create_webdataset(my_dataset_dir, output_pattern, max_shard_size=1e9) # 1e9 bytes = 1 GB
        print("\nWebDataset creation completed successfully.")
    except Exception as e:
        print(f"\nError during WebDataset creation: {e}")
        logging.error(f"Error during WebDataset creation: {e}")
