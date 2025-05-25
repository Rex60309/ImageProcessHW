def wavelet_transform_step(data):
    """
    Apply one step of wavelet transform

    Args:
        data: List of numbers to transform

    Returns:
        tuple: (low_pass_coefficients, high_pass_coefficients)
    """
    if len(data) % 2 != 0:
        raise ValueError("Data length must be even")

    low_pass = []  # s coefficients (averages)
    high_pass = []  # d coefficients (differences)

    # Process pairs of adjacent elements
    for i in range(0, len(data), 2):
        a = data[i]
        b = data[i + 1]

        # Low pass: s = (a + b) / 2
        s = (a + b) / 2
        low_pass.append(s)

        # High pass: d = a - s
        d = a - s
        high_pass.append(d)

    return low_pass, high_pass

def full_wavelet_transform(data):
    """
    Apply complete wavelet transform until we get single coefficient

    Args:
        data: List of numbers to transform

    Returns:
        dict: Contains all levels of transformation
    """
    result = {'original': data}
    current_data = data[:]
    level = 1

    print(f"Original data: {data}")
    print("-" * 50)

    while len(current_data) > 1:
        low_pass, high_pass = wavelet_transform_step(current_data)

        print(f"Level {level}:")
        print(f"  Low pass (s): {low_pass}")
        print(f"  High pass (d): {high_pass}")

        result[f'level_{level}'] = {
            'low_pass': low_pass,
            'high_pass': high_pass
        }

        current_data = low_pass
        level += 1

    return result

def inverse_wavelet_step(low_pass, high_pass):
    """
    Reconstruct data from one level of wavelet coefficients

    Args:
        low_pass: List of s coefficients
        high_pass: List of d coefficients

    Returns:
        list: Reconstructed data
    """
    if len(low_pass) != len(high_pass):
        raise ValueError("Low pass and high pass must have same length")

    reconstructed = []

    for i in range(len(low_pass)):
        s = low_pass[i]
        d = high_pass[i]

        # Solve: s = (a + b) / 2 and d = a - s
        # From d = a - s, we get a = d + s
        # From s = (a + b) / 2, we get b = 2s - a = 2s - (d + s) = s - d
        a = d + s
        b = s - d

        reconstructed.extend([a, b])

    return reconstructed

def full_inverse_wavelet(transform_result):
    """
    Reconstruct original data from complete wavelet transform

    Args:
        transform_result: Dictionary containing wavelet transform levels

    Returns:
        list: Reconstructed original data
    """
    print("\nReconstructing original sequence:")
    print("-" * 50)

    # Find the highest level
    levels = [key for key in transform_result.keys() if key.startswith('level_')]
    max_level = max([int(level.split('_')[1]) for level in levels])

    # Start from the highest level (single coefficient)
    current_low = transform_result[f'level_{max_level}']['low_pass']

    # Reconstruct level by level
    for level in range(max_level, 0, -1):
        high_pass = transform_result[f'level_{level}']['high_pass']

        print(f"Level {level} reconstruction:")
        print(f"  Input low pass: {current_low}")
        print(f"  Input high pass: {high_pass}")

        current_low = inverse_wavelet_step(current_low, high_pass)

        print(f"  Reconstructed: {current_low}")

    return current_low

def solve_problem(data, problem_label):
    """
    Solve one complete problem (forward and inverse transform)

    Args:
        data: Input data sequence
        problem_label: Label for the problem (e.g., "Problem A", "Problem B")
    """
    print(f"\n{'='*60}")
    print(f"{problem_label}")
    print(f"{'='*60}")

    # Part (i): Apply wavelet transform
    print(f"\n(i) Applying wavelet transform to {data}")
    transform_result = full_wavelet_transform(data)

    # Part (ii): Recover the input sequence
    print(f"\n(ii) Recovering the input sequence")
    recovered = full_inverse_wavelet(transform_result)

    print(f"\nOriginal sequence:  {data}")
    print(f"Recovered sequence: {recovered}")

    # Verify reconstruction
    if data == recovered:
        print("✓ Reconstruction successful!")
    else:
        print("✗ Reconstruction failed!")

    return transform_result, recovered

def main():
    """Main function to solve both problems"""

    # Problem A
    data_a = [71, 67, 24, 26, 36, 32, 14, 18]
    solve_problem(data_a, "Problem A")

    # Problem B
    data_b = [18, 14, 32, 36, 26, 24, 67, 71]
    solve_problem(data_b, "Problem B")

if __name__ == "__main__":
    main()