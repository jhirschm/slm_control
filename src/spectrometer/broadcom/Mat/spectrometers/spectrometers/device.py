"""Functions to find and initialize a device"""

import copy

from spectrometers import devices


def find_devices():
    """Find all devices"""

    device_object_list = []
    device_id_list = []
    device_coordinate_list = []
    for device_class_index, device_class in enumerate(
        devices.device_classes.values()
    ):
        device_object_list.append(device_class())
        device_object_list[-1].initialize()
        for device_id_index, device_id in enumerate(
            device_object_list[-1].device_id_list
        ):
            device_id_list.append(device_id)
            device_coordinate_list.append(
                [device_class_index, device_id_index]
            )

    return device_coordinate_list, device_id_list, device_object_list


def choose_device(
    device_object_list,
    device_coordinates,
):
    """Choose a device from the list"""

    try:
        device_object = device_object_list[device_coordinates[0]]
        device_object.device_index = device_coordinates[1]
    except (IndexError, ValueError) as exc:
        raise ValueError("Device not found.") from exc

    return device_object


def find_choose_device():
    """Find devices and choose a device from the list"""

    device_coordinate_list, device_id_list, device_object_list = find_devices()

    if len(device_coordinate_list) > 1:
        print("Found devices:")
        for device_id_index, device_id in enumerate(device_id_list):
            print("Index " + str(device_id_index) + ": " + device_id)
        device_index = input("Enter index of device: ")
        try:
            device_coordinates = device_coordinate_list[int(device_index)]
        except (IndexError, ValueError) as exc:
            raise ValueError("Device not found.") from exc
    elif len(device_coordinate_list) == 1:
        device_coordinates = device_coordinate_list[0]
    else:
        raise ValueError("Device not found.")

    return choose_device(device_object_list, device_coordinates)
