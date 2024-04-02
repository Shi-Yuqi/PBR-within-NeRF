import taichi as ti  

@ti.data_oriented
class ParticleSimulator:  
    def __init__(self, n_particles=1000, dt=1e-4, gravity=[0, -9.8], window_res=(512, 512)):  
        ti.init(arch=ti.cpu)  
          
        self.n_particles = n_particles  
        self.dt = dt  
        self.gravity = gravity  
        self.window_res = window_res  
          
        self.pos = ti.Vector.field(2, dtype=float, shape=n_particles)  
        self.vel = ti.Vector.field(2, dtype=float, shape=n_particles)  
        self.acc = ti.Vector.field(2, dtype=float, shape=n_particles)  
          
        self.gui = ti.GUI('下落场景', res=window_res)  
          
        self.initialize()  
      
    @ti.kernel  
    def initialize(self):  
        for i in range(self.n_particles):  
            self.pos[i] = [ti.random(), ti.random()]  
            self.vel[i] = [0, 0]  
            self.acc[i] = self.gravity  
      
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
              
            self.advance()  
              
            self.gui.circles(self.pos.to_numpy(), radius=1, color=0x6495ED)  
            self.gui.show()  
  
# 使用示例  
simulator = ParticleSimulator()  
simulator.run_simulation()  