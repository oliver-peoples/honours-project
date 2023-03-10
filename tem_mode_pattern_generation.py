from scipy.special import hermite as physicistsHermite
import math

print(physicistsHermite(3)(1))

def temModeFnXY(m: int, n: int, x: float, y: float, I_0: float = 1, w_0: float = 1, w: float = 1) -> float:
    
    return I_0 * (w_0 / w)**2 * (physicistsHermite(m)((2**0.5 * x)/w) * math.exp(-x**2 / w**2))**2 * (physicistsHermite(n)((2**0.5 * y)/w) * math.exp(-y**2 / w**2))**2

def main():
    
    
    

if __name__ == "__main__":
    
    main()