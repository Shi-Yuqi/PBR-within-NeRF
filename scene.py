import taichi as ti

@ti.data_oriented
class FallenParticle():
    def __init__(self, n_particles, dt, gravity, circle_radius, window_res):

        self.n_particles = n_particles
        self.dt = dt
        self.gravity = gravity
        self.circle_radius = circle_radius
        self.window_res = window_res

        self.pos = ti.Vector.field(2, dtype=float, shape=n_particles)
        self.vel = ti.Vector.field(2, dtype=float, shape=n_particles)
        self.acc = ti.Vector.field(2, dtype=float, shape=n_particles)

        self.gui = ti.GUI("scene", res=window_res)
        self.initialize()

    # @ti.kernel
    # def initialize(self):
    #     for i in range(self.n_particles):
    #         self.pos[i] = [ti.random(), ti.random()]
    #         self.vel[i] = [0, 0]
    #         self.acc[i] = self.gravity

    @ti.kernel
    def initialize(self):
        i = 0
        while(i < self.n_particles):
            self.pos[i] = [ti.random(), ti.random()]
            flag = 0
            j = 0
            while(j < i):
                distance = (self.pos[i] - self.pos[j]).norm()
                if distance < 2 * self.circle_radius:
                    flag = 1
                    break
                j = j + 1
            if flag == 0:
                self.vel[i] = [0, 0]
                self.acc[i] = self.gravity
                i = i + 1

    @ti.kernel
    def advance(self):
        for i in range(self.n_particles):
            self.vel[i] += self.dt * self.acc[i]
            self.pos[i] += self.dt * self.vel[i]
    
    def run_simulation(self):
        while self.gui.running:
            for e in self.gui.get_events(ti.GUI.PRESS):
                if e.key == ti.GUI.ESCAPE:
                    self.gui.running = False
        
            self.advance()  # 更新粒子状态  
        
            self.gui.circles(self.pos.to_numpy(), radius=self.circle_radius, color=0x6495ED)  # 在窗口中绘制粒子
            self.gui.show()
# 定义仿真参数
