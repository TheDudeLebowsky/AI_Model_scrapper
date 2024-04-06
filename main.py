import logging
import time
import os
import subprocess
from pprint import pformat
from huggingface_hub import scan_cache_dir, snapshot_download
from colorama import Fore, Style
from huggingface_hub import list_datasets
logging.basicConfig(level=logging.INFO)
from datasets import get_dataset_config_names, get_dataset_infos
import re


class AIScrapper():
    def init(self):
        self.delay_between_download = 1 #seconds

        self.API_FILE = "api.txt"
        self.USE_TOKEN = False
        self.cache_info_file = "cache_info.txt"
        self.repository_list = "repository_list.txt"
        self.repository_downloaded = "repository_downloaded.txt"
        self.all_datasets = "all_datasets.txt"
        self.datasets_list = "datasets_list.txt"
        self.datasets_downloaded = "datasets_downloaded.txt"
        self.custom_cache_dir = 'D:\models\huggingface'
        self.suffix_complete = " DONE"
        self.suffix_failed = " ERROR : please verify if item is publicly available"
    
    def run_command(self, command):
        """ Executes a given command in the terminal and returns its output. """

        try:
            # Execute the command and capture the output
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            # If an error occurs, return the error message
            return f"An error occurred: {e.stderr}"


    #===========================================================HF DOWNLOAD UTILS==========================================================#

    def huggingface_download(self, DOWNLOAD_FROM_LIST, item_name, use_auth_token=None, local_dir=None, resume_download=True, token=None, item_type="Model"):
        """
        Download a repository from Hugging Face and optionally initialize a model and tokenizer.

        Args:
        item_name (str): Identifier of the model or dataset on Hugging Face (e.g., 'username/item-name').
        use_auth_token (str, optional): Authentication token for Hugging Face
        local_dir (str, optional): Local directory to store the downloaded files.

        # Download the snapshot
        #for faster download : pip installhf_transfer
        #HF_HUB_ENABLE_HF_TRANSFER=1 as an environment variable.    
        
        """
        if item_type == "Model":
            input_file = self.repository_list
            output_file = self.repository_downloaded

            config_names = None
        elif item_type == "Dataset":
            input_file = self.datasets_list
            output_file = self.datasets_downloaded

        if item_name == '':
            item_name = f'All items from list {input_file}'
        print(f"\n{Fore.LIGHTWHITE_EX}{Style.BRIGHT}Downloading {item_type} '{item_name}'{Style.RESET_ALL}\n")

        if DOWNLOAD_FROM_LIST == True:
            self.download_from_list(input_file, output_file, local_dir, use_auth_token, self.custom_cache_dir, resume_download, token, item_type)
        else:        
            self.download_single_item(item_name,input_file, output_file, local_dir, use_auth_token, self.custom_cache_dir, resume_download, token, item_type)
            
            
            

    def download_single_item(self, item_name, input_file, output_file, local_dir, use_auth_token, Cache_dir, Resume_download, Token, item_type):

        self.print_download_info(item_name, use_auth_token, local_dir, cache_dir=Cache_dir, resume_download=Resume_download, token=Token)
        
        with open(input_file, 'r') as file, open(output_file, 'a') as completed_file:
            for line in file:
                item_name = line.strip()
                if not item_name:
                    completed_file.write(line)  # Write the unmodified line to the temp file
                    continue  # Skip empty lines
            
            if item_type == "Model":
                try:
                    item_path = snapshot_download(item_name, local_dir=local_dir, cache_dir=Cache_dir, resume_download=Resume_download, token=Token)
                except Exception as e:
                    print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                    self.success_write2file(output_file, item_name, self.suffix_failed)
                    exit()
            
            elif item_type == "Dataset":
                print(f'\nFetching config names...')
                try:

                    snapshot_download(repo_id=item_name,repo_type="dataset", local_dir=local_dir, cache_dir=Cache_dir, resume_download=Resume_download, token=Token)  
                except Exception as e:
                    print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                    self.success_write2file(output_file, item_name, self.suffix_failed)
                    exit()
                item_path = Cache_dir + "\\" + item_name
            
        
        
        summary = "\n"
        summary = f"{Fore.LIGHTGREEN_EX}{Style.BRIGHT}TASK COMPLETED{Style.RESET_ALL}\n"
        summary += f"{item_type} : {Fore.LIGHTCYAN_EX}{Style.BRIGHT}{item_name}{Style.RESET_ALL}\n"
        summary += f"Downloaded at : {Fore.LIGHTCYAN_EX}{Style.BRIGHT}{item_path}{Style.RESET_ALL}\n"
        summary += f"Saved to {Fore.LIGHTCYAN_EX}{Style.BRIGHT}{self.repository_downloaded}{Style.RESET_ALL}\n"
        print(summary)
        self.success_write2file(output_file, item_name, self.suffix_complete)
        time.sleep(self.delay_between_download)

    def download_from_list(self, input_file, output_file, local_dir, use_auth_token, Cache_dir, Resume_download, Token, item_type):
        if not os.path.exists(input_file):
            print(f"{Fore.RED}Error: {input_file} file not found.{Style.RESET_ALL}")
            return

        with open(input_file, 'r') as file, open(output_file, 'a') as completed_file:
            for line in file:
                item_name = line.strip()
                if not item_name:
                    completed_file.write(line)  # Write the unmodified line to the temp file
                    continue  # Skip empty lines
                
                self.print_download_info(item_name, use_auth_token, local_dir, cache_dir=Cache_dir, resume_download=Resume_download, token=Token)

                if item_type == "Model":
                    try:
                        item_path = snapshot_download(item_name, local_dir=local_dir, cache_dir=Cache_dir, resume_download=Resume_download, token=Token)
                    except Exception as e:
                        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                        self.success_write2file(output_file, item_name, self.suffix_failed)
                        continue
                elif item_type == "Dataset":


                    try:    
                        snapshot_download(repo_id=item_name,repo_type="dataset", local_dir=local_dir, cache_dir=Cache_dir, resume_download=Resume_download, token=Token)  
            
                    except Exception as e:
                        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                        self.success_write2file(output_file, item_name, self.suffix_failed)
                        continue
                item_path = Cache_dir + "\\" + item_name
                
                
                summary = ""
                summary = f"\n{Fore.LIGHTGREEN_EX}{Style.BRIGHT}TASK COMPLETED{Style.RESET_ALL}\n"
                summary += f"{item_type} : {Fore.LIGHTCYAN_EX}{Style.BRIGHT}{item_name}{Style.RESET_ALL}\n"
                summary += f"Downloaded at : {Fore.LIGHTCYAN_EX}{Style.BRIGHT}{item_path}{Style.RESET_ALL}\n"
                summary += f"Saved to {Fore.LIGHTCYAN_EX}{Style.BRIGHT}{self.repository_downloaded}{Style.RESET_ALL}\n"
                print(summary)
                self.success_write2file(output_file, item_name, self.suffix_complete) 
                time.sleep(self.delay_between_download)

    #==============================================================DATASETS===============================================================#
    def all_datasets_list(self, slow=True):
        """
        Get a list of all available datasets.
        Save the list to a file.
        Slow mode : wait 2 seconds x datasets
        """

        counter = 0
        print(f"\n{Fore.LIGHTWHITE_EX}{Style.BRIGHT}Available datasets:\n{Style.RESET_ALL}")
        with open(self.all_datasets, "a") as file:
            for dataset in list_datasets():
                print(f"{dataset.id}")
                file.write(dataset.id + "\n")

                counter += 1
                if slow and counter >= 1000:
                    print(f"{Fore.LIGHTYELLOW_EX}{Style.BRIGHT}Waiting 2 seconds{Style.RESET_ALL}")
                    time.sleep(2)
                    counter = 0

                    file.flush() 
        
    def print_dataset_info(dataset_name):
        """
        Prints information about a dataset.
        for each configuration, print the description, homepage and citation.
        
        

        Args:
            dataset_name (_type_): _description_
        """

        config_names = get_dataset_config_names(dataset_name)
        dataset_info = "\n"
        dataset_info += f"Dataset : {Fore.LIGHTMAGENTA_EX}{Style.BRIGHT}{dataset_name}{Style.RESET_ALL}\n"
        dataset_info += f"\nAvailable configurations:{config_names}{Style.RESET_ALL}\n"
        print(dataset_info)

        # Get dataset information for each configuration
        for config_name in config_names:
            info = get_dataset_infos(dataset_name)[config_name]
            config_info = "\n"
            config_info += f"\nInfo for {Fore.LIGHTCYAN_EX}{Style.BRIGHT}{config_name}{Style.RESET_ALL}:\n"
            config_info += f"\nDescription:\n{Fore.LIGHTCYAN_EX}{info.description}{Style.RESET_ALL}"
            config_info += f"Homepage: {Fore.LIGHTCYAN_EX}{info.homepage}{Style.RESET_ALL}\n"
            #config_info += f"Citation: {info.citation}\n"

            print(config_info)

    def print_download_info(item_name, use_auth_token=None, local_dir=None, cache_dir=None, resume_download=False, token=None):
        """
        Prints summary of tasks
        
        """
        
        downloading = ""
        if resume_download==True:
            downloading += f"{Fore.LIGHTGREEN_EX}{Style.BRIGHT}RESUMING DOWNLOAD{Style.RESET_ALL}\n"
        else:
            downloading += f"{Fore.LIGHTGREEN_EX}{Style.BRIGHT}STARTING DOWNLOAD{Style.RESET_ALL}\n"
        downloading += f"Item name : {Fore.LIGHTCYAN_EX}{Style.BRIGHT}{item_name}{Style.RESET_ALL}\n"
        if use_auth_token is not None:    
            downloading += f"Using auth : {Fore.LIGHTCYAN_EX}{Style.BRIGHT}{use_auth_token}{Style.RESET_ALL}\n"
            
        if local_dir is not None:
            downloading += f"local dir : {Fore.LIGHTCYAN_EX}{Style.BRIGHT}{local_dir}{Style.RESET_ALL}\n"
        if cache_dir is not None:
            downloading += f"cache fir : {Fore.LIGHTCYAN_EX}{Style.BRIGHT}{cache_dir}{Style.RESET_ALL}\n"
        if token is not None:
            downloading += f"token : {Fore.LIGHTCYAN_EX}{Style.BRIGHT}{token}{Style.RESET_ALL}"
        print(downloading)
        time.sleep(2)

    #============================================================GENERAL UTILS============================================================#

    def get_api_token(self):   
        """
        Get the API token from a file.
        """
        filename = self.API_FILE
        with open(filename, 'r') as file:
            ACCESS_TOKEN = file.readline().strip()
            print(f"ACCESS token : {Fore.GREEN}{Style.BRIGHT}{ACCESS_TOKEN}{Style.RESET_ALL}")
            return ACCESS_TOKEN

    def huggingface_scan_cache(self):  #NEED TO TEST
        """
        to delete need pip install huggingface_hub["cli"]
        Use : 
        huggingface-cli delete-cache --dir D:\models\huggingface

        """


        scan_cache_command=f"huggingface-cli delete-cache --dir {self.custom_cache_dir}"
        print(f"scan_cache_command : {Fore.LIGHTCYAN_EX}{Style.BRIGHT}{scan_cache_command}{Style.RESET_ALL}")
        try:
            hf_cache_info = scan_cache_dir(self.custom_cache_dir)
            pretty_hf_cache_info = pformat(hf_cache_info)
            print(pretty_hf_cache_info)

            output = self.run_command(scan_cache_command)
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
            output = f"{Fore.RED}Error: {e}{Style.RESET_ALL}"
        return output


    def success_write2file(self, output_file, item_name, suffix):
        lines_changed = False

        # Regular expression pattern to match the item_name followed by a space or end of line
        pattern = re.compile(r'\b' + re.escape(item_name) + r'\b')

        with open(output_file, 'r') as completed_file:
            # Read all lines in the file
            lines = completed_file.readlines()

        # Process each line and replace if item_name is found
        for i in range(len(lines)):
            if pattern.search(lines[i]):
                # Entry found, print the message
                print(f"{Fore.LIGHTYELLOW_EX}{Style.BRIGHT}Entry for {item_name} already in file")
                print(f"{lines[i]}")
                print(f"Updating...{Style.RESET_ALL}\n")

                # Preserve the original line ending
                line_ending = lines[i][-1]
                lines[i] = item_name + suffix + line_ending
                lines_changed = True
                break

        # Write back the modified content to the file
        if lines_changed:
            with open(output_file, 'w') as completed_file:
                completed_file.writelines(lines)
        else:
            with open(output_file, 'a') as completed_file:
                completed_file.write(item_name + suffix + '\n')




    def get_config_names(self, item_name):
            config_names = get_dataset_config_names(item_name)
            print(f"config_names : {Fore.LIGHTCYAN_EX}{Style.BRIGHT}{config_names}{Style.RESET_ALL}\n")
            return config_names










def main():
    scrapper = AIScrapper()
    if scrapper.USE_TOKEN == True:
        Token = scrapper.get_api_token()
    else:
        Token=None

    #OPTIONAL :
  
    #all_datasets_list(slow=True) #Uncomment to save all datasets ID to a file

    #print_dataset_info("indonli") #Uncomment to print a dataset info

    #cache = huggingface_scan_cache() 

    

    DOWNLOAD_FROM_LIST = False
    item_Type = input(f"\n{Fore.LIGHTWHITE_EX}{Style.BRIGHT}Enter item type (Model/Dataset): {Style.RESET_ALL}")
    
    if item_Type != "Model" and item_Type != "Dataset":
        print(f"{Fore.RED}Error: Invalid item type.{Style.RESET_ALL}")
        return
    
    item_name = input(f"\n{Fore.LIGHTWHITE_EX}{Style.BRIGHT}Enter item name: (Skip to continue downloading from list){Style.RESET_ALL}")    
    if item_name == "":
        DOWNLOAD_FROM_LIST = True
    scrapper.huggingface_download(DOWNLOAD_FROM_LIST, item_name, local_dir=None, resume_download=True, item_type=item_Type, token=Token)          
    input(f"{Fore.LIGHTYELLOW_EX}{Style.BRIGHT}Press any key to continue{Style.RESET_ALL}")
    exit()


if __name__ == '__main__':

    main()  
