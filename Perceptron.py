"""A perceptron or Threshold Logic Unit (TLU) is an artificial
neural network, which takes inputs with weights, and computes output
based on step function chosen, and sum of weigts multiplied by the
inputs. In this program I model TLU as a binary classifier."""

def perceptron(x, w, t):
    """Method to model a perceptron functioning"""
    output = sum([x[i] * w[i] for i in range(len(x))])
    if output >= t:
        print("Positive class")
    else:
        print("Negative class")
    return

if __name__ == '__main__':
    print("Please enter entries as space separated values: ")
    x = list(map(float, input("Enter inputs: ").split()))
    w = list(map(float, input("Enter weights: ").split()))
    t = float(input("Enter threshold: "))
    perceptron(x, w, t)
