import subprocess
import os

def get_device_for_path(path):
    """Get the device that a path resides on."""
    try:
        result = subprocess.run(['df', path, '--output=source'], capture_output=True, text=True, check=True)
        # The output includes headers, so we split by lines and return the second line
        return result.stdout.splitlines()[1].strip()
    except subprocess.CalledProcessError as e:
        print(f"Error finding device for path {path}: {e}")
        return None

def check_nvme_support_for_path(path):
    try:
        device = get_device_for_path(path)
        if device:
            # Use 'lsblk' to list details of the device and check if it's NVMe
            result = subprocess.run(['lsblk', device, '-d', '-o', 'name,tran'], capture_output=True, text=True, check=True)
            if 'nvme' in result.stdout:
                print(f"The path {path} resides on an NVMe disk.")
                return True
            else:
                print(f"The path {path} does not reside on an NVMe disk.")
                return False
        else:
            print("Could not determine the device for the given path.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error checking NVMe support for path {path}: {e}")
        return False

if __name__ == "__main__":
    path = '.'  
    check_nvme_support_for_path(path)