
from collections import defaultdict
from random import randint

# Bucket Sort
# Time:  O(n + klogk) ~ O(n + nlogn)
# Space: O(n)
class BucketSort(object):
    def topKFrequent(self, words, k):
        counts = defaultdict(int)
        for ws in words:
            for w in ws:
                counts[w] += 1

        buckets = [[]] * (sum(counts.values()) + 1)
        for i, count in counts.items():
            buckets[count].append(i)

        result = []
        # result_append = result.append
        for i in reversed(range(len(buckets))):
            for j in range(len(buckets[i])):
                # slower
                # result_append(buckets[i][j])
                result.append(buckets[i][j])
                if len(result) == k:
                    return result
        return result


# Quick Select
# Time:  O(n) ~ O(n^2), O(n) on average.
# Space: O(n)
class QuickSelect(object):
    def topKFrequent(self, words, k):
        """
        :type words: List[str]
        :type k: int
        :rtype: List[str]
        """
        counts = defaultdict(int)
        for ws in words:
            for w in ws:
                counts[w] += 1
        p = []
        for key, val in counts.items():
            p.append((-val, key))
        self.kthElement(p, k)

        result = []
        sorted_p = sorted(p[:k])
        for i in range(k):
            result.append(sorted_p[i][1])
        return result

    def kthElement(self, nums, k):  # O(n) on average
        def PartitionAroundPivot(left, right, pivot_idx, nums):
            pivot_value = nums[pivot_idx]
            new_pivot_idx = left
            nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
            for i in range(left, right):
                if nums[i] < pivot_value:
                    nums[i], nums[new_pivot_idx] = nums[new_pivot_idx], nums[i]
                    new_pivot_idx += 1

            nums[right], nums[new_pivot_idx] = nums[new_pivot_idx], nums[right]
            return new_pivot_idx

        left, right = 0, len(nums) - 1
        while left <= right:
            pivot_idx = randint(left, right)
            new_pivot_idx = PartitionAroundPivot(left, right, pivot_idx, nums)
            if new_pivot_idx == k - 1:
                return
            elif new_pivot_idx > k - 1:
                right = new_pivot_idx - 1
            else:  # new_pivot_idx < k - 1.
                left = new_pivot_idx + 1


# top_k_selector = BucketSort()

top_k_selector = QuickSelect()
