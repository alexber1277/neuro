package neuro

import (
	"log"
	"sort"
	"sync"
)

type Genetic struct {
	Nets    []*NetPerc  `json:"nets"`
	Config  GeneticConf `json:"conf"`
	Percent float64     `json:"percent"`
	Error   float64     `json:"error"`
}

type GeneticConf struct {
	Population     int          `json:"population"`
	LastBest       int          `json:"last_best"`
	MinRandWeight  float64      `json:"min_weight"`
	MaxRandWeight  float64      `json:"max_weight"`
	BestResult     float64      `json:"best_result"`
	LimitMutateSub int          `json:"limit_mutate_sub"`
	Data           []*DataTeach `json:"data"`
}

var defaultConf = GeneticConf{
	Population:     500,
	LastBest:       100,
	MinRandWeight:  -100,
	MaxRandWeight:  100,
	LimitMutateSub: 100,
	BestResult:     0.001,
}

func InitGenetic(conf ...GeneticConf) *Genetic {
	g := &Genetic{Config: defaultConf}
	if len(conf) != 0 {
		g.Config = conf[0]
	}
	g.Error = 1
	return g
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
	g.Nets = append(g.Nets, net)
	return g
}

func (g *Genetic) sortBest() *Genetic {
	sort.Slice(g.Nets, func(i, j int) bool {
		return g.Nets[i].Error < g.Nets[j].Error
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

	g.Percent = g.GetBest().Result.Percent
	g.Error = g.GetBest().Error

	if !last {
		g.mutate()
	}
}

func (g *Genetic) sliceBest() {
	g.Nets = g.Nets[:g.Config.LastBest]
}

func (g *Genetic) mutate() {
	var (
		wg          sync.WaitGroup
		listNetsAdd []*NetPerc
	)
	ind := len(g.Nets) - 1
	for s := ind; s <= g.Config.Population; s++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r := randInt(len(g.Nets) - 1)
			n := g.Nets[r].Copy()
			var wgw sync.WaitGroup
			for i := 0; i < g.Config.LimitMutateSub; i++ {
				wgw.Add(1)
				go func() {
					wgw.Done()
					n.mutateWeight(g.Config.MinRandWeight, g.Config.MaxRandWeight)
				}()
			}
			wgw.Wait()
			listNetsAdd = append(listNetsAdd, n)
		}()
	}
	wg.Wait()
	g.Nets = append(g.Nets, listNetsAdd...)
}
