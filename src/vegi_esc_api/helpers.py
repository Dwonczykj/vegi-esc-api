import re


def parse_measurement(measurement_str: str):
    pattern = r'(\d+)(\w+)'
    match = re.match(pattern, measurement_str)

    if match:
        number = int(match.group(1))
        unit = match.group(2)
        return number, unit
    else:
        return None, None


# Example usage
measurement = "1kg"
result = parse_measurement(measurement)
if result:
    number, unit = result
    print("Number of units:", number)
    print("Unit type:", unit)
else:
    print("Invalid measurement format.")
