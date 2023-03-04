from evaluating import evaluate as e

if __name__ == '__main__':
    # Name method
    method = 'VGG16'
    # Create Image Search System object
    IS = e(method)
    # Evaluating system
    IS.evaluating()
    print("Done VGG16")

    method = 'Xception'
    # Create Image Search System object
    IS = e(method)
    # Evaluating system
    IS.evaluating()
    print("Done Xception")
    
    method = 'EfficientNetV2L'
    # Create Image Search System object
    IS = e(method)
    # Evaluating system
    IS.evaluating()
    print("Done EfficientNetV2L")