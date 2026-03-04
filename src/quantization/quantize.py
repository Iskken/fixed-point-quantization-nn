import numpy as np

def fixed_point_quantize(float_array, total_bits=8, fractional_bits=4):
    """
    Simulates fixed-point quantization.
    
    1. Scales the float by 2^fractional_bits.
    2. Rounds to the nearest integer.
    3. Clamps values to fit within the bit-width range.
    4. Scales back to float for simulation compatibility.
    """
    
    #Calculate the range of the hardware 
    min_val = -(2**(total_bits - 1))
    max_val = 2**(total_bits - 1) - 1
    
    # Scaling factor
    scaling_factor = 2**fractional_bits
    
    # 3. Quantization Step
    # We multiply by the factor, then round
    quantized_scaled = np.round(float_array * scaling_factor)
    
    # 4. Clamping (Handling Overflow)
    # If a number is too big for the bits, it "saturates" at the max/min
    clamped = np.clip(quantized_scaled, min_val, max_val)
    
    # 5. De-quantization (Back to Float)
    # This represents the "noisy" version of the original number
    # To be able to run it on our machine
    float_quantized = clamped / scaling_factor
    
    return float_quantized

def calculate_quantization_error(original, quantized):
    """
    Observe how the loss/weights change.
    This calculates the 'Quantization Noise'.
    """
    return np.mean((original - quantized)**2)