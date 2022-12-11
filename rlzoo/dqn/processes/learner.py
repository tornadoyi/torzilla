from rlzoo.zoo.role import Role


class Learner(Role):
    def _run(self):
        self.manager().learner.gear.connect(
            lambda method, *args, **kwargs: getattr(self, method)(*args, **kwargs)
        ).join()

    def close(self):
        self.manager().learner.gear.close()