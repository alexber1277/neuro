package neuro

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"sort"
	"sync"
	"time"
)

type Genetic struct {
	Nets    []*NetPerc  `json:"nets"`
	Config  GeneticConf `json:"conf"`
	Percent float64     `json:"percent"`
	Error   float64     `json:"error"`
	Score   float64     `json:"score"`
	Iters   int         `json:"iters"`
	Tm      time.Time   `json:"tm"`
}

type GeneticConf struct {
	Population     int          `json:"population"`
	LastBest       int          `json:"last_best"`
	MinRandWeight  float64      `json:"min_weight"`
	MaxRandWeight  float64      `json:"max_weight"`
	BestResult     float64      `json:"best_result"`
	LimitMutateSub int          `json:"limit_mutate_sub"`
	Budget         float64      `json:"budget"`
	Data           []*DataTeach `json:"data"`
}

var defaultConf = GeneticConf{
	Population:     500,
	LastBest:       100,
	MinRandWeight:  -100,
	MaxRandWeight:  100,
	LimitMutateSub: 100,
	BestResult:     0.001,
	Budget:         1000,
}

func InitGenetic(conf ...GeneticConf) *Genetic {
	g := &Genetic{Config: defaultConf}
	g.Tm = time.Now()
	if len(conf) != 0 {
		g.Config = conf[0]
	}
	g.Error = 1
	return g
}

func (g *Genetic) GetTm() string {
	return time.Now().Sub(g.Tm).String()
}

func (g *Genetic) GetBest() *NetPerc {
	return g.Nets[0]
}

func (g *Genetic) Add(ret func() *NetPerc) {
	for i := 0; i < g.Config.Population; i++ {
		g.AddNet(ret())
	}
}

func (g *Genetic) SetData(data []*DataTeach) *Genetic {
	g.Config.Data = data
	return g
}

func (g *Genetic) AddNet(net *NetPerc) *Genetic {
	net.Budget = g.Config.Budget
	g.Nets = append(g.Nets, net)
	return g
}

func (g *Genetic) ClearScore() {
	g.Score = 0
}

func (g *Genetic) sortBest() *Genetic {
	sort.Slice(g.Nets, func(i, j int) bool {
		g.Nets[i].Score = g.Nets[i].DiffPerce * float64(g.Nets[i].Trades)
		g.Nets[j].Score = g.Nets[j].DiffPerce * float64(g.Nets[j].Trades)
		return g.Nets[i].Score > g.Nets[j].Score
	})
	return g
}

func (g *Genetic) sortBestNols() *Genetic {
	sort.Slice(g.Nets, func(i, j int) bool {
		return g.Nets[i].Nols < g.Nets[j].Nols
	})
	return g
}

func (g *Genetic) showList(max ...int) {
	for i, el := range g.Nets {
		if len(max) > 0 && max[0] >= i {
			log.Println(i, el.Result)
		}
	}
}

func (g *Genetic) TrainItem(ret func(n *NetPerc)) {
	var wg sync.WaitGroup
	for i, _ := range g.Nets {
		wg.Add(1)
		go func(ind int) {
			defer wg.Done()
			g.Nets[ind].Trades = 0
			g.Nets[ind].Nols = 0
			g.Nets[ind].Score = 0
			g.Nets[ind].Budget = g.Config.Budget
			g.Nets[ind].LastPrice = 0
			g.Nets[ind].DiffPerce = 0
			g.Nets[ind].StatusBSell = false
			ret(g.Nets[ind])
		}(i)
	}
	wg.Wait()
}

func (g *Genetic) Train(last bool) {
	var wg sync.WaitGroup
	wg.Add(len(g.Nets))
	for i, _ := range g.Nets {
		go func(ind int) {
			defer wg.Done()
			g.Nets[ind].TrainIter()
		}(i)
	}
	wg.Wait()

	g.sortBest()
	g.sliceBest()

	if !last {
		g.mutate()
	}

	g.Score = g.GetBest().Score
}

func (g *Genetic) CheckScore() bool {
	return g.Score >= g.Config.BestResult
}

func (g *Genetic) LogScore(i int) {
	if g.Iters%i == 0 {
		log.Println(
			g.Iters, " - iter; ",
			"MAX:", toFixed(g.Nets[0].Budget, 3),
			"MIN:", toFixed(g.Nets[len(g.Nets)-1].Budget, 3), "; ",
			"len:", len(g.Nets), "; ",
			"score:", fmt.Sprintf("%.3f", g.GetBest().Score), "; ",
			"trades:", g.GetBest().Trades, "; ",
			"diff:", toFixed(g.GetBest().DiffPerce, 3),
		)
	}
}

func (g *Genetic) LogScoreNols(i int) {
	if g.Iters%i == 0 {
		log.Println(g.Iters, " - iter; ", g.Nets[0].Nols, " - nols")
	}
}

func (g *Genetic) sortBestNolsIters() {

}

func (g *Genetic) Iterate() bool {
	g.sortBest()
	g.sliceBest()
	g.Score = g.GetBest().Budget
	g.mutate()
	g.Iters += 1
	return false

}

func (g *Genetic) IterateNols(lastNols int) bool {
	g.sortBestNols()
	g.sliceBest()
	var stats int
	for i := 0; i < lastNols; i++ {
		if g.Nets[i].Nols == 0 {
			stats += 1
		}
	}
	if stats == lastNols {
		return true
	}
	g.Iters += 1
	g.mutate()
	return false
}

func (g *Genetic) sliceBest() {
	if len(g.Nets) > g.Config.LastBest {
		g.Nets = g.Nets[:g.Config.LastBest]
	}
}

func (g *Genetic) mutate() {
	var (
		wg          sync.WaitGroup
		mtWait      sync.Mutex
		listNetsAdd []*NetPerc
	)
	ind := len(g.Nets) - 1
	if ind == 0 {
		for s := ind; s < g.Config.Population; s++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				n := g.Nets[0].Copy()
				var wgw sync.WaitGroup
				for i := 0; i < g.Config.LimitMutateSub; i++ {
					wgw.Add(1)
					go func(nn *NetPerc) {
						defer wgw.Done()
						nn.mutateWeight(g.Config.MinRandWeight, g.Config.MaxRandWeight)
					}(n)
				}
				wgw.Wait()
				mtWait.Lock()
				listNetsAdd = append(listNetsAdd, n)
				mtWait.Unlock()

			}()
		}
	} else {
		for s := ind; s < g.Config.Population; s++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				r := randIntMin(1, ind)
				n := g.Nets[r].Copy()
				var wgw sync.WaitGroup
				for i := 0; i < g.Config.LimitMutateSub; i++ {
					wgw.Add(1)
					go func(nn *NetPerc) {
						defer wgw.Done()
						nn.mutateWeight(g.Config.MinRandWeight, g.Config.MaxRandWeight)
					}(n)
				}
				wgw.Wait()
				mtWait.Lock()
				listNetsAdd = append(listNetsAdd, n)
				mtWait.Unlock()

			}()
		}
	}
	wg.Wait()
	g.Nets = append(g.Nets, listNetsAdd...)
}

func (g *Genetic) Save(fileName string) error {
	if fileName == "" {
		return errors.New("empty filename")
	}
	if bts, err := json.Marshal(g); err != nil {
		return err
	} else {
		if err := ioutil.WriteFile(fileName, bts, 0644); err != nil {
			return err
		}
	}
	return nil
}

func (g *Genetic) Load(fileName string) bool {
	if fileName == "" {
		log.Println("empty filename")
		return false
	}
	bts, err := ioutil.ReadFile(fileName)
	if err != nil {
		log.Println(err)
		return false
	}
	if err := json.Unmarshal(bts, &g); err != nil {
		log.Println(err)
		return false
	}
	return true
}
