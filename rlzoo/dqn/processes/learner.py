from rlzoo.zoo.role import Role


class Learner(Role):
    def _run(self):

        self.manager().learner_gear.connect(
            lambda method, *args, **kwargs: getattr(self, method)(*args, **kwargs)
        ).join()