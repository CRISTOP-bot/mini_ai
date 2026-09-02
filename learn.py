import argparse,subprocess,sys
from research import run

def main():
 p=argparse.ArgumentParser(); p.add_argument('--max-queries',type=int,default=20); p.add_argument('--max-steps',type=int,default=5000); a=p.parse_args()
 class R: pass
 r=R(); r.queries='research_queries.txt'; r.out='dataset/research'; r.max_queries=a.max_queries; r.max_pages=2; r.max_bytes=2000000; r.max_text_per_source=12000; r.min_text=200
 run(r); subprocess.run([sys.executable,'train.py','--auto','--max-steps',str(a.max_steps)],check=True)
if __name__=='__main__': main()
