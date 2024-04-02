from scene import FallenParticle
import taichi as ti

ti.init(arch=ti.gpu)

def main():
    n_particles = 10
    dt = 1e-4
    gravity = [0, -98.1]
    circle_radius = 10
    window_res = (512, 512)

    scene = FallenParticle(
        n_particles=n_particles,
        dt=dt,
        gravity=gravity,
        circle_radius=circle_radius,
        window_res=window_res,
    )
    scene.run_simulation()

if __name__ == "__main__":
    main()