import random
import time
from collections import Counter
from tqdm import tqdm
from art import tprint
import os
import numpy as np
import tabulate


def ordinal(n: int):
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix

clear = lambda: os.system("clear")

algorithms = [
    "heapsort",
    "ipmergesort",
    "mergesort",
    "treesort",
    "blocksort",
    "smoothsort",
    "timesort",
    "patiencesort",
    "cubesort",
    "quicksort",
    "sheelsort",
    "compsort",
    "cocktailsort",
    "gnomesort",
    "pancake_sort",
    "strandsort",
    "selectionsort",
    "exchangesort",
    "cyclesort",
    "bucketsort",
    "pigeonholesort",
    "lsdradixsort"
]



def heapsort (array):
    def heapify(array, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2

        if l < n and array[i] < array[l]:
            largest = l

        if r < n and array[largest] < array[r]:
            largest = r

        if largest != i:
            array[i], array[largest] = array[largest], array[i]
            heapify(array, n, largest)

    n = len(array)

    for i in range(n // 2 - 1, -1, -1):
        heapify(array, n, i)

    for i in range(n - 1, 0, -1):
        array[i], array[0] = array[0], array[i]
        heapify(array, i, 0)

    return array

def ipmergesort(array):
    def merge(left, right):
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

        result.extend(left[i:])
        result.extend(right[j:])
        return result

    if len(array) <= 1:
        return array

    mid = len(array) // 2
    left = ipmergesort(array[:mid])
    right = ipmergesort(array[mid:])

    return merge(left, right)

def mergesort(array):
    def merge(left, right):
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

        result.extend(left[i:])
        result.extend(right[j:])
        return result

    if len(array) <= 1:
        return array

    mid = len(array) // 2
    left = mergesort(array[:mid])
    right = mergesort(array[mid:])

    return merge(left, right)

def tournementsort(array):
    def tournament(arr):
        tree = [None] * 2 * (len(arr) + len(arr) % 2)
        index = len(tree) - len(arr)
        tree[index: index + len(arr)] = arr
        for i in range(index - 1, -1, -1):
            tree[i] = min(tree[2 * i + 1], tree[2 * i + 2])
        return tree

    def update(tree, index, value):
        tree[index] = value
        index = (index - 1) // 2
        while index >= 0:
            tree[index] = min(tree[2 * index + 1], tree[2 * index + 2])
            index = (index - 1) // 2

    def get_min(tree):
        return tree[0]

    n = len(array)
    tree = tournament(array)
    sorted_array = []
    for _ in range(n):
        min_val = get_min(tree)
        sorted_array.append(min_val)
        array[array.index(min_val)] = float("inf")
        update(tree, tree.index(min_val) + len(tree) // 2, float("inf"))
    return sorted_array

def treesort(array):
    class Node:
        def __init__(self, key):
            self.key = key
            self.left = None
            self.right = None

    def insert(node, key):
        if node is None:
            return Node(key)
        if key < node.key:
            node.left = insert(node.left, key)
        else:
            node.right = insert(node.right, key)
        return node

    def store_sorted(node, sorted_list):
        if node is not None:
            store_sorted(node.left, sorted_list)
            sorted_list.append(node.key)
            store_sorted(node.right, sorted_list)

    root = None
    for i in array:
        root = insert(root, i)

    sorted_array = []
    store_sorted(root, sorted_array)
    return sorted_array

def blocksort(array):
    def insertion_sort(arr):
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and key < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr

    n = len(array)
    m = 1 << 8
    block_size = int(n ** 0.5)
    blocks = [array[i: i + block_size] for i in range(0, n, block_size)]
    for i in range(len(blocks)):
        blocks[i] = insertion_sort(blocks[i])
    sorted_array = []
    for i in range(0, n, block_size):
        sorted_array += blocks[i // block_size]
    return insertion_sort(sorted_array)

def smoothsort(array):
    def sift_down(array, l, r, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < l and array[i] < array[left]:
            largest = left

        if right < r and array[largest] < array[right]:
            largest = right

        if largest != i:
            array[i], array[largest] = array[largest], array[i]
            sift_down(array, l, r, largest)

    def heapify(array):
        n = len(array)
        p = 1
        while p < n:
            p = 3 * p + 1
        while p > 0:
            for i in range(p, n):
                sift_down(array, n, i + 1, i - p)
            p //= 3

    n = len(array)
    heapify(array)
    for i in range(n - 1, 0, -1):
        array[0], array[i] = array[i], array[0]
        sift_down(array, i, 0, 0)
    return array

def timesort(array):
    def insertion_sort(arr, left, right):
        for i in range(left + 1, right + 1):
            key_item = arr[i]
            j = i - 1
            while j >= left and arr[j] > key_item:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key_item
        return arr

    def merge(arr, l, m, r):
        len1, len2 = m - l + 1, r - m
        left, right = arr[l: m + 1], arr[m + 1: r + 1]
        i, j, k = 0, 0, l
        while i < len1 and j < len2:
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
        while i < len1:
            arr[k] = left[i]
            i += 1
            k += 1
        while j < len2:
            arr[k] = right[j]
            j += 1
            k += 1

    def tim_sort(arr):
        n = len(arr)
        min_run = 32
        for i in range(0, n, min_run):
            insertion_sort(arr, i, min((i + min_run - 1), n - 1))
        size = min_run
        while size < n:
            for left in range(0, n, 2 * size):
                mid = min(n - 1, left + size - 1)
                right = min((left + 2 * size - 1), (n - 1))
                merge(arr, left, mid, right)
            size = 2 * size
        return arr

    return tim_sort(array)

def patiencesort(array):
    def binary_search(tails, key):
        l, r = 0, len(tails) - 1
        while l <= r:
            m = (l + r) // 2
            if tails[m] == key:
                return m
            elif tails[m] < key:
                l = m + 1
            else:
                r = m - 1
        return l

    tails = []
    for i in array:
        pos = binary_search(tails, i)
        if pos == len(tails):
            tails.append(i)
        else:
            tails[pos] = i
    return tails

def cubesort(array):
    def cube_sort(arr):
        n = len(arr)
        gap = n // 2
        while gap > 0:
            for i in range(gap, n):
                temp = arr[i]
                j = i
                while j >= gap and arr[j - gap] > temp:
                    arr[j] = arr[j - gap]
                    j -= gap
                arr[j] = temp
            gap //= 2
        return arr

    return cube_sort(array)

def quicksort(array):
    def partition(arr, low, high):
        i = low - 1
        pivot = arr[high]
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    def quick_sort(arr, low, high):
        if low < high:
            pi = partition(arr, low, high)
            quick_sort(arr, low, pi - 1)
            quick_sort(arr, pi + 1, high)
        return arr

    return quick_sort(array, 0, len(array) - 1)

def sheelsort(array):
    def shell_sort(arr):
        n = len(arr)
        gap = n // 2
        while gap > 0:
            for i in range(gap, n):
                temp = arr[i]
                j = i
                while j >= gap and arr[j - gap] > temp:
                    arr[j] = arr[j - gap]
                    j -= gap
                arr[j] = temp
            gap //= 2
        return arr

    return shell_sort(array)

def compsort(array):
    def comb_sort(arr):
        n = len(arr)
        gap = n
        shrink = 1.3
        sorted_ = False
        while not sorted_:
            gap = int(gap / shrink)
            if gap <= 1:
                gap = 1
                sorted_ = True
            i = 0
            while i + gap < n:
                if arr[i] > arr[i + gap]:
                    arr[i], arr[i + gap] = arr[i + gap], arr[i]
                    sorted_ = False
                i += 1
        return arr

    return comb_sort(array)

def cocktailsort(array):
    def cocktail_sort(arr):
        n = len(arr)
        swapped = True
        start = 0
        end = n - 1
        while swapped:
            swapped = False
            for i in range(start, end):
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    swapped = True
            if not swapped:
                break
            swapped = False
            end -= 1
            for i in range(end - 1, start - 1, -1):
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    swapped = True
            start += 1
        return arr

    return cocktail_sort(array)

def gnomesort(array):
    def gnome_sort(arr):
        n = len(arr)
        index = 0
        while index < n:
            if index == 0:
                index = 1
            if arr[index] >= arr[index - 1]:
                index += 1
            else:
                arr[index], arr[index - 1] = arr[index - 1], arr[index]
                index -= 1
        return arr

    return gnome_sort(array)

def odd_even_sort(array):
    def odd_even_sort(arr):
        n = len(arr)
        is_sorted = 0
        while is_sorted == 0:
            is_sorted = 1
            for i in range(1, n - 1, 2):
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    is_sorted = 0
            for i in range(0, n - 1, 2):
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    is_sorted = 0
        return arr

    return odd_even_sort(array)

def pancake_sort(array):
    def flip(arr, i):
        start = 0
        while start < i:
            arr[start], arr[i] = arr[i], arr[start]
            start += 1
            i -= 1

    def find_max(arr, n):
        mi = 0
        for i in range(0, n):
            if arr[i] > arr[mi]:
                mi = i
        return mi

    def pancake_sort(arr):
        n = len(arr)
        curr_size = n
        while curr_size > 1:
            mi = find_max(arr, curr_size)
            if mi != curr_size - 1:
                flip(arr, mi)
                flip(arr, curr_size - 1)
            curr_size -= 1
        return arr

    return pancake_sort(array)

def strandsort(array):
    def merge(left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    def strand_sort(arr):
        result = []
        while arr:
            sublist = [arr.pop(0)]
            i = 0
            while i < len(arr):
                if arr[i] > sublist[-1]:
                    sublist.append(arr.pop(i))
                else:
                    i += 1
            result = merge(result, sublist)
        return result

    return strand_sort(array)

def selectionsort(array):
    def selection_sort(arr):
        n = len(arr)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return arr

    return selection_sort(array)

def exchangesort(array):
    def exchange_sort(arr):
        n = len(arr)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if arr[i] > arr[j]:
                    arr[i], arr[j] = arr[j], arr[i]
        return arr

    return exchange_sort(array)

def cyclesort(array):
    def cycle_sort(arr):
        n = len(arr)
        for cycle_start in range(n - 1):
            item = arr[cycle_start]
            pos = cycle_start
            for i in range(cycle_start + 1, n):
                if arr[i] < item:
                    pos += 1
            if pos == cycle_start:
                continue
            while item == arr[pos]:
                pos += 1
            arr[pos], item = item, arr[pos]
            while pos != cycle_start:
                pos = cycle_start
                for i in range(cycle_start + 1, n):
                    if arr[i] < item:
                        pos += 1
                while item == arr[pos]:
                    pos += 1
                arr[pos], item = item, arr[pos]
        return arr

    return cycle_sort(array)

def bucketsort(input_list):
    def insertion_sort(bucket):
        for i in range(1, len(bucket)):
            var = bucket[i]
            j = i - 1
            while (j >= 0 and var < bucket[j]):
                bucket[j + 1] = bucket[j]
                j = j - 1
            bucket[j + 1] = var
        return bucket
    # Find maximum value in the list and use length of the list to determine which value in the list goes into which bucket 
    max_value = max(input_list)
    size = max_value/len(input_list)

    # Create n empty buckets where n is equal to the length of the input list
    buckets_list= []
    for x in range(len(input_list)):
        buckets_list.append([]) 

    # Put list elements into different buckets based on the size
    for i in range(len(input_list)):
        j = int (input_list[i] / size)
        if j != len (input_list):
            buckets_list[j].append(input_list[i])
        else:
            buckets_list[len(input_list) - 1].append(input_list[i])

    # Sort elements within the buckets using Insertion Sort
    for z in range(len(input_list)):
        insertion_sort(buckets_list[z])
            
    # Concatenate buckets with sorted elements into a single list
    final_output = []
    for x in range(len (input_list)):
        final_output = final_output + buckets_list[x]
    return final_output

def pigeonholesort(array):
    def pigeonhole_sort(arr):
        n = len(arr)
        min_ = min(arr)
        max_ = max(arr)
        size = max_ - min_ + 1
        holes = [0] * size
        for x in arr:
            holes[x - min_] += 1
        i = 0
        for count in range(size):
            while holes[count] > 0:
                holes[count] -= 1
                arr[i] = count + min_
                i += 1
        return arr

    return pigeonhole_sort(array)

def lsdradixsort(array):
    def counting_sort(arr, exp):
        n = len(arr)
        output = [0] * n
        count = [0] * 10
        for i in range(n):
            index = arr[i] // exp
            count[index % 10] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]
        i = n - 1
        while i >= 0:
            index = arr[i] // exp
            output[count[index % 10] - 1] = arr[i]
            count[index % 10] -= 1
            i -= 1
        i = 0
        for i in range(n):
            arr[i] = output[i]

    def radix_sort(arr):
        max_ = max(arr)
        exp = 1
        while max_ // exp > 0:
            counting_sort(arr, exp)
            exp *= 10
        return arr

    return radix_sort(array)

    
    
    
    
    
def randomize_algo_list(algo): #randomize the list of algorithms
    for i in range(len(algo)):
        j = random.randint(0, len(algo) - 1)
        algo[i], algo[j] = algo[j], algo[i]
    return algo

def run_sorting_algorithm(algo, array): #run the asked algorithm
    if algo == "heapsort":
        return heapsort(array)
    elif algo == "ipmergesort":
        return ipmergesort(array)
    elif algo == "mergesort":
        return mergesort(array)
    elif algo == "tournementsort":
        return tournementsort(array)
    elif algo == "treesort":
        return treesort(array)
    elif algo == "blocksort":
        return blocksort(array)
    elif algo == "smoothsort":
        return smoothsort(array)
    elif algo == "timesort":
        return timesort(array)
    elif algo == "patiencesort":
        return patiencesort(array)
    elif algo == "cubesort":
        return cubesort(array)
    elif algo == "quicksort":
        return quicksort(array)
    elif algo == "sheelsort":
        return sheelsort(array)
    elif algo == "compsort":
        return compsort(array)
    elif algo == "cocktailsort":
        return cocktailsort(array)
    elif algo == "gnomesort":
        return gnomesort(array)
    elif algo == "odd-even_sort":
        return odd_even_sort(array)
    elif algo == "pancake_sort":
        return pancake_sort(array)
    elif algo == "strandsort":
        return strandsort(array)
    elif algo == "selectionsort":
        return selectionsort(array)
    elif algo == "exchangesort":
        return exchangesort(array)
    elif algo == "cyclesort":
        return cyclesort(array)
    elif algo == "bucketsort":
        return bucketsort(array)
    elif algo == "pigeonholesort":
        return pigeonholesort(array)
    elif algo == "lsdradixsort":
        return lsdradixsort(array)
    else:
        return None   
    
def compare_algo(algo1, algo2, len_arr): #take two algorithms and return their execution time
    arr = generate_random_array(len_arr)
    arr1 = arr.copy()
    arr2 = arr.copy()
    # compare execution time
    start = time.time()
    run_sorting_algorithm(algo1, arr1)
    end = time.time()
    time1 = end - start
    start = time.time()
    run_sorting_algorithm(algo2, arr2)
    end = time.time()
    time2 = end - start
    return time1, time2

def generate_random_array(length):
    arr= list(range(length))
    return random.sample(arr, length)

timetook = {}
for i in algorithms:
    timetook[i] = []
    
def compare_all_algos():
    winners = {}
    classment = []
    algorithmslist = randomize_algo_list(algorithms)

    while len(algorithmslist) > 1:
        for i in range(len(algorithmslist) - 1):
            time1, time2 = compare_algo(algorithmslist[i], algorithmslist[i + 1], 10)
            # print("COMPARE", algorithmslist[i], "AND", algorithmslist[i + 1], end="\r", flush=True)
            if time1 < time2:
                winners[algorithmslist[i]] = winners.get(algorithmslist[i], 0) + 1
                timetook[algorithmslist[i]].append(time1*1000)
                classment.append(algorithmslist[i+1])
            else:
                winners[algorithmslist[i + 1]] = winners.get(algorithmslist[i + 1], 0) + 1
                timetook[algorithmslist[i+1]].append(time2*1000)
                classment.append(algorithmslist[i])
        algorithmslist = list(winners.keys())
        winners = {}
    # invert the classment
    return algorithmslist[0]

def final_winner(itterations=10000):
    
    winners = []
    for i in tqdm(range(itterations)):
        winners.append(compare_all_algos())
    count = Counter(winners).most_common()
    print(count)
    clear()
    tprint("Results")
    table = [["Place","Name", "Itterations", "Percentage", "AVG Time"]]
    for i in range(len(count)):
        avg = np.mean(timetook[count[i][0]])
        table.append([ordinal(i+1), count[i][0], count[i][1], str(round(count[i][1]/itterations*100,2))+"%", np.format_float_positional(avg, trim='-')+"ms"])
            
    print(tabulate.tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    
    # secondmost_common = 
    # end(most_common, Counter(winners).most_common(1)[0][1], itterations, Counter(winners).most_common(1)[0][1]/itterations*100)
    # print("The most common algorithm is:", most_common, "with a frequency of", Counter(winners).most_common(1)[0][1], "out of", itterations, "wich represent", Counter(winners).most_common(1)[0][1]/itterations*100, "% of the time.")
  
def home():
    clear()
    tprint('Algorythms Olympics')
    print("Press enter to continue")
    input("")
    clear()
    tprint("Options")
    itterations = input("Select How many itterations you want (10 000 by default):")
    clear()
    tprint("Calculating...")
    if itterations != "":
        final_winner(int(itterations))
    else:
        final_winner()
        
def end(most_common, frequency, itt, percent):
    tprint("Winner")
    print('The most common algorithm is:', most_common, 'with a frequency of', frequency, 'out of', itt, 'which represent', percent, "% of the time")
    
    
    
    
home()



