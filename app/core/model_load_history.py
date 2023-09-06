class History:
    def __init__(self, size):
        self._size = size
        self._data = []

    @property
    def data(self):
        """
        查看当前加载历史记录
        按照加载时间排序 越靠前 代表越靠近当前时间点
        :return:
        """
        return self._data[::-1]

    def add(self, name):
        if name in self._data:
            self._data.remove(name)
        self._data.append(name)
        if len(self._data) > self._size:
            self._data = self._data[len(self._data) - self._size:]

    def is_exist(self, name):
        """
        返回当前模型所在的索引
        值越小，代表最近使用的时间越靠近当前
        值为-1，代表最近没有使用过
        :param name:
        :return:
        """
        if name in self.data:
            return self.data.index(name)
        return -1
