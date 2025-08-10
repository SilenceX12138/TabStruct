from abc import abstractmethod


class BasePipeline:

    @classmethod
    def run(cls, args):
        return cls._run(args)

    @classmethod
    @abstractmethod
    def _run(cls, args):
        raise NotImplementedError("This method has to be implemented by the sub class")
