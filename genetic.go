package neuro

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/thoas/go-funk"
)

type Genetic struct {
	Nets      []*NetPerc  `json:"nets"`
	ResOrders []*ResOrder `json:"res_orders"`
	Config    GeneticConf `json:"conf"`
	Percent   float64     `json:"percent"`
	Error     float64     `json:"error"`
	Score     float64     `json:"score"`
	Iters     int         `json:"iters"`
	LBOitem   LBO         `json:"lbo_item"`
	Tm        time.Time   `json:"tm"`
}

type GeneticConf struct {
	Population     int          `json:"population"`
	LastBest       int          `json:"last_best"`
	MinRandWeight  float64      `json:"min_weight"`
	MaxRandWeight  float64      `json:"max_weight"`
	BestResult     float64      `json:"best_result"`
	LimitMutateSub int          `json:"limit_mutate_sub"`
	Budget         float64      `json:"budget"`
	TradesByDay    float64      `json:"trades_by_day"`
	Inps           int          `json:"inps"`
	NewItems       int          `json:"new_items"`
	Hours          float64      `json:"hours"`
	MinPerce       float64      `json:"min_perce"`
	Data           []*DataTeach `json:"data"`
}

type LBO struct {
	Score  float64 `json:"score"`
	Count  int     `json:"count"`
	Trades []int   `json:"trades"`
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

type ResOrder struct {
	Count  int     `json:"count"`
	By     int     `json:"by"`
	Type   bool    `json:"type"`
	Sum    float64 `json:"sum"`
	Score  float64 `json:"score"`
	Trades []int   `json:"trades"`
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

func (g *Genetic) GenerateTrades() *Genetic {
	for i := 0; i < g.Config.Population; i++ {
		g.ResOrders = append(g.ResOrders, &ResOrder{Trades: []int{}})
	}
	return g
}

func (g *Genetic) GenerateTradesV2() *Genetic {
	for i := 0; i < g.Config.Inps; i++ {
		g.ResOrders = append(g.ResOrders, &ResOrder{Count: i + 1, Trades: []int{}})
	}
	return g
}

func (g *Genetic) GenerateOrdersV2() *Genetic {
	for _, r := range g.ResOrders {
		for t := 0; t < r.Count; t++ {
			r.Trades = append(r.Trades, t)
		}
	}
	return g
}

func (g *Genetic) FirstMutate(inpData []DataTeach) *Genetic {
	g.LBOitem = LBO{}
	for {
		var wg sync.WaitGroup
		wg.Add(len(g.ResOrders))
		for _, ord := range g.ResOrders {
			func(ordItem *ResOrder) {
				defer wg.Done()
				ordItem.Sum = g.Config.Budget
				ordItem.Type = false
				lastPrice := 0.0
				for _, pr := range ordItem.Trades {
					if !ordItem.Type {
						ordItem.Sum -= inpData[pr].Price
						ordItem.Type = true
						lastPrice = inpData[pr].Price
					} else {
						if getDiff(inpData[pr].Price, lastPrice) > g.Config.MinPerce {
							ordItem.Sum += inpData[pr].Price
							ordItem.Type = false
						}
					}
				}
			}(ord)
		}
		wg.Wait()
		g.sortBestOrders()
		if g.Score < g.GetBestOrders().Score {
			g.LBOitem.Count = g.GetBestOrders().Count
			g.LBOitem.Score = g.GetBestOrders().Score
			copy(g.LBOitem.Trades, g.GetBestOrders().Trades)
			g.Score = g.GetBestOrders().Score
			g.LogScoreOrders(1)
		} else {
			//g.LogScoreOrders(1000)
		}

		g.mutateOrdersFirst()
		g.Iters += 1
	}

	return g
}

func (g *Genetic) GenerateOrders(maxLength int) *Genetic {
	for _, r := range g.ResOrders {
		r.Count = randIntMin(1, int(maxLength/2))
		r.Trades = []int{}
		r.By = int(math.Floor(float64(maxLength / r.Count)))
		for i := 0; i < maxLength-1; i++ {
			if i%r.By == 0 {
				r.Trades = append(r.Trades, i)
			}
		}
		r.Trades = r.Trades[:r.Count]
	}
	return g
}

func (g *Genetic) AddOrder() *ResOrder {
	r := ResOrder{}
	r.Count = randIntMin(1, int(g.Config.Inps))
	r.Trades = []int{}
	r.By = int(math.Floor(float64(g.Config.Inps / r.Count)))
	for i := 0; i < g.Config.Inps-1; i++ {
		if i%r.By == 0 {
			r.Trades = append(r.Trades, i)
		}
	}
	r.Trades = r.Trades[:r.Count]
	return &r
}

func (g *Genetic) GetTm() string {
	return time.Now().Sub(g.Tm).String()
}

func (g *Genetic) GetBest() *NetPerc {
	return g.Nets[0]
}

func (g *Genetic) GetBestOrders() *ResOrder {
	return g.ResOrders[0]
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
		g.Nets[i].Score = g.Nets[i].DiffPerce / (g.Config.Hours/24*g.Config.TradesByDay + float64(g.Nets[i].Trades))
		g.Nets[j].Score = g.Nets[j].DiffPerce / (g.Config.Hours/24*g.Config.TradesByDay + float64(g.Nets[j].Trades))
		return g.Nets[i].Score > g.Nets[j].Score
	})
	return g
}

func (g *Genetic) sortBestOrders() *Genetic {
	sort.Slice(g.ResOrders, func(i, j int) bool {
		g.ResOrders[i].Score = g.ResOrders[i].Sum
		g.ResOrders[j].Score = g.ResOrders[j].Sum
		// ============================
		//g.ResOrders[i].Score = float64(g.ResOrders[i].Count) / g.ResOrders[i].Sum
		//g.ResOrders[j].Score = float64(g.ResOrders[j].Count) / g.ResOrders[j].Sum
		// ============================
		//g.ResOrders[i].Score = (g.ResOrders[i].Sum / 10000) / (float64(g.ResOrders[i].Count) * 0.9)
		//g.ResOrders[j].Score = (g.ResOrders[j].Sum / 10000) / (float64(g.ResOrders[j].Count) * 0.9)
		return g.ResOrders[i].Score > g.ResOrders[j].Score
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

func (g *Genetic) TrainItemOrders(ret func(r *ResOrder)) {
	var wg sync.WaitGroup
	for i, _ := range g.ResOrders {
		wg.Add(1)
		go func(ind int) {
			defer wg.Done()
			g.ResOrders[ind].Score = 0
			g.ResOrders[ind].Sum = g.Config.Budget
			ret(g.ResOrders[ind])
		}(i)
	}
	wg.Wait()
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

func (g *Genetic) LogScoreOrders(i int) {
	if g.Iters%i == 0 {
		log.Println(
			g.Iters, " - iter; ",
			"SCORE:", g.GetBestOrders().Score, "; ",
			"COUNT:", g.GetBestOrders().Count, "; ",
			"SUM:", g.GetBestOrders().Sum, "; ",
			"BY:", g.GetBestOrders().By, "; ",
			"LENGTH:", len(g.ResOrders), "; ",
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
	//g.mutate()
	g.mutateV2()
	g.Iters += 1
	return false
}

func (g *Genetic) IterateOrders() {
	g.sortBestOrders()
	g.sliceBestOrders()
	g.Score = g.GetBestOrders().Sum
	g.mutateOrders()
	g.Iters += 1
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

func (g *Genetic) sliceBestOrders() {
	if len(g.ResOrders) > g.Config.LastBest {
		g.ResOrders = g.ResOrders[:g.Config.LastBest]
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

func (g *Genetic) mutateV2() {
	var (
		wg          sync.WaitGroup
		mtWait      sync.Mutex
		listNetsAdd []*NetPerc
	)
	ind := len(g.Nets) - 1
	for s := ind; s < g.Config.Population; s++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			n := g.Nets[0].Copy()
			n.mutateWeight(g.Config.MinRandWeight, g.Config.MaxRandWeight)
			mtWait.Lock()
			listNetsAdd = append(listNetsAdd, n)
			mtWait.Unlock()
		}()
	}
	wg.Wait()
	g.Nets = append(g.Nets, listNetsAdd...)
}

func (g *Genetic) mutateOrders() {
	var (
		wg         sync.WaitGroup
		mtWait     sync.Mutex
		listOrders []*ResOrder
	)
	ind := len(g.ResOrders) - 1
	for s := ind; s < g.Config.Population; s++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rnd := randIntMin(0, ind)
			r := g.ResOrders[rnd].Copy()
			r.mutate()
			mtWait.Lock()
			listOrders = append(listOrders, r)
			mtWait.Unlock()
		}()
	}
	wg.Wait()
	for i := 0; i < g.Config.NewItems; i++ {
		g.ResOrders = append(g.ResOrders, g.AddOrder())
	}
	g.ResOrders = append(g.ResOrders, listOrders...)
}

func (r *ResOrder) Copy() *ResOrder {
	rr := ResOrder{
		Count: r.Count,
		By:    r.By,
		Type:  r.Type,
		Sum:   r.Sum,
		Score: r.Score,
	}
	rr.Trades = make([]int, len(r.Trades))
	copy(rr.Trades, r.Trades)
	return &rr
}

func remove(s []int, i int) []int {
	s[i] = s[len(s)-1]
	return s[:len(s)-1]
}

func (r *ResOrder) mutateAll(max int) {
	mp := make(map[int]struct{})
	trds := []int{}
	for {
		in := funk.RandomInt(0, max)
		if _, ok := mp[in]; ok {
			continue
		}
		mp[in] = struct{}{}
		trds = append(trds, in)
		if len(trds) >= r.Count {
			break
		}
	}
	sort.Slice(trds, func(i, j int) bool {
		return trds[i] < trds[j]
	})
	r.Trades = trds
}

func (r *ResOrder) mutate() {
	var nVal int
	rand := randIntMin(0, len(r.Trades)-1)
	if rand == 0 {
		if len(r.Trades) != 1 {
			nVal = randIntMin(0, r.Trades[rand+1]-1)
		}
	}
	if rand == len(r.Trades)-1 {
		if len(r.Trades) != 1 {
			nVal = randIntMin(r.Trades[rand-1], r.Trades[rand])
		}
	}
	if rand != 0 && rand != len(r.Trades)-1 {
		if len(r.Trades) != 1 {
			nVal = randIntMin(r.Trades[rand-1], r.Trades[rand+1])
		}
	}
	r.Trades[rand] = nVal
}

func (g *Genetic) mutateOrdersFirst() {
	var wg sync.WaitGroup
	wg.Add(len(g.ResOrders))
	for _, ord := range g.ResOrders {
		func(ordItem *ResOrder) {
			defer wg.Done()
			ordItem.mutateAll(g.Config.Inps)
		}(ord)
	}
	wg.Wait()
}
