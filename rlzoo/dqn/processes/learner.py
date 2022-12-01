from rlzoo.zoo.role import Role


class Learner(Role):
    def _on_start(self, *args, **kwargs):

        self.manager.learner_gear.connect(
            lambda method, *args, **kwargs: getattr(self, method)(*args, **kwargs)
        ).join()