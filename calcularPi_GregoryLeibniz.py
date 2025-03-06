import multiprocessing as mp
import time
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np

def gregory_leibniz_worker(start: int, end: int) -> float:
    """Calculate partial sum of Gregory-Leibniz series from start to end indices."""
    partial_sum = 0
    for i in range(start, end):
        partial_sum += (-1)**i / (2*i + 1)
    return partial_sum

def calculate_pi_parallel(num_processes: int, time_limit: int = 60) -> Tuple[int, float]:
    """
    Calculate pi using parallel processes for a given time limit.
    Returns the number of iterations completed and the calculated pi value.
    """
    chunk_size = 1000000  # Process chunks of iterations at a time
    total_iterations = 0
    pi_value = 0
    start_time = time.time()
    
    with mp.Pool(processes=num_processes) as pool:
        while (time.time() - start_time) < time_limit:
            # Calculate chunk ranges for each process
            ranges = [(i * chunk_size, (i + 1) * chunk_size) 
                     for i in range(num_processes)]
            
            # Calculate partial sums in parallel
            results = pool.starmap(gregory_leibniz_worker, ranges)
            
            # Sum results and update total
            pi_value = sum(results) * 4
            total_iterations += chunk_size * num_processes
            
            if (time.time() - start_time) >= time_limit:
                break
    
    return total_iterations, pi_value

def main():
    process_counts = range(1, 21)  # 1 to 20 processes
    results: List[Tuple[int, float]] = []
    
    print("Running calculations for different process counts...")
    print("Process Count | Iterations | Pi Value | Time (s)")
    print("-" * 50)
    
    for n_processes in process_counts:
        start_time = time.time()
        iterations, pi_value = calculate_pi_parallel(n_processes)
        elapsed_time = time.time() - start_time
        
        results.append((iterations, pi_value))
        print(f"{n_processes:12d} | {iterations:10d} | {pi_value:.10f} | {elapsed_time:.2f}")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.plot(list(process_counts), [r[0] for r in results], 'b-o')
    plt.grid(True)
    plt.xlabel('Number of Processes')
    plt.ylabel('Iterations Completed in 60 seconds')
    plt.title('Performance Scaling of Pi Calculation')
    
    # Add actual values as annotations
    for i, (iterations, _) in enumerate(results):
        plt.annotate(f'{iterations:,}', 
                    (process_counts[i], iterations),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')
    
    plt.tight_layout()
    plt.savefig('pi_calculation_performance.png')
    print("\nResults have been plotted to 'pi_calculation_performance.png'")

if __name__ == '__main__':
    main()