from ase.io.trajectory import Trajectory
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=None, help='parallel workers', required=True)
    args = parser.parse_args()

    qgbr = Trajectory("QGBR.traj", mode='a')
    for i in range(args.num_workers):
        traji = Trajectory(f"QGBR_{i}.traj")
        print(f"{len(traji)} atoms in QGBR_{i}")
        for ats in traji:
            qgbr.write(ats, append=True)
    qgbr.close()
    print(f"total atoms: {len(qgbr)}")