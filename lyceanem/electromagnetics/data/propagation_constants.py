from importlib_resources import files

import lyceanem.electromagnetics.data as data


def oxygen_lines():
    data_lines = []
    oxy_data = str(files(data).joinpath("Oxy.txt"))
    with open(oxy_data, "r") as file:
        for line in file:
            if line.strip():
                values = [float(x) for x in line.split()]
                data_lines.append(values[:7])

    return data_lines


def water_vapour_lines():

    data_lines = []
    water_data = str(files(data).joinpath("Vapour.txt"))
    with open(water_data, "r") as file:
        for line in file:
            if line.strip():
                values = [float(x) for x in line.split()]
                data_lines.append(values[:7])

    return data_lines
