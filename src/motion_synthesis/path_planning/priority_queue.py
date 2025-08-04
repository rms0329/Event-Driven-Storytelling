from heapq import heappop, heappush


class RemovedSentinel:
    def __lt__(self, other):
        return True

    def __gt__(self, other):
        return False


_REMOVED = RemovedSentinel()


class PriorityQueue:
    """
    Min heap with priority update available. Lower priority value has higher priority.
    (ref: https://docs.python.org/3/library/heapq.html)
    """

    def __init__(self, item=None, priority=None) -> None:
        self.lst = []
        self.finder = {}
        if item is not None and priority is not None:
            self.push(item, priority=priority)

    def __contains__(self, item):
        return item in self.finder

    def push(self, item, priority):
        assert item not in self.finder, f"'{item}' has already been pushed!"
        entry = [priority, item]
        self.finder[item] = entry
        heappush(self.lst, entry)

    def update(self, item, priority):
        assert item in self.finder, f"'{item}' has not been pushed!"
        self.remove(item)
        self.push(item, priority)

    def remove(self, item):
        assert item in self.finder, f"'{item}' has not been pushed!"
        entry = self.finder.pop(item)
        entry[-1] = _REMOVED

    def pop(self):
        while self.lst:
            priority, item = heappop(self.lst)
            if item is not _REMOVED:
                del self.finder[item]
                return item
        raise KeyError("pop from empty priority queue")

    def is_empty(self):
        return not self.finder


if __name__ == "__main__":
    pq = PriorityQueue()
    pq.push(4, priority=4)
    pq.push(6, priority=6)
    pq.push(3, priority=3)
    pq.push(7, priority=7)
    pq.push(1, priority=1)
    pq.push(2, priority=2)
    pq.push(5, priority=5)

    pq.update(7, priority=0)
    pq.update(5, priority=-1)
    pq.remove(1)

    print(pq.pop())
    print(pq.pop())
    print(pq.pop())
    print(pq.pop())
    print(pq.pop())

    pq.push(8, priority=8)
    pq.push(1, priority=1)

    while not pq.is_empty():
        print(pq.pop())
