x = 3
print(type(x)) 

xs = [3, 1, 2] 
print(xs[2])

nums = list(range(5)) # range sinh ta 1 list các phần tử
print(nums) # Prints "[0, 1, 2, 3, 4]"
print(nums[2:4]) 

animals = ['cat', 'dog', 'monkey']
# duyệt giá trị không cần chỉ số
for animal in animals:
    print('%s' % (animal))
# duyệt giá trị kèm chỉ số dùng enumerate
for key, value in enumerate(animals):
    print('#%d: %s' % (key + 1, value))
