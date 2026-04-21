import random
import copy
import numpy as np
import gymnasium as gym 
import os
from multiprocessing import Process, Queue

# CONFIG
ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = 'human'
TEST_EPISODES = 1000
STEPS = 500

NUM_PROCESSES = os.cpu_count()
evaluationQueue = Queue()
evaluatedQueue = Queue()


nInputs = 8
nOutputs = 2
SHAPE = (nInputs,12,nOutputs)
GENOTYPE_SIZE = 0
for i in range(1, len(SHAPE)):
    GENOTYPE_SIZE += SHAPE[i-1]*SHAPE[i]

POPULATION_SIZE = 100
NUMBER_OF_GENERATIONS = 100
PROB_CROSSOVER = 0.9

PROB_MUTATION = 1.0/GENOTYPE_SIZE
STD_DEV = 0.1


ELITE_SIZE = 1

def network(shape, observation,ind):
    #Computes the output of the neural network given the observation and the genotype
    x = observation[:]
    for i in range(1,len(shape)):
        y = np.zeros(shape[i])
        for j in range(shape[i]):
            for k in range(len(x)):
                y[j] += x[k]*ind[k+j*len(x)]
        x = np.tanh(y)
    return x

def check_successful_landing(observation):
    #Checks the success of the landing based on the observation
    x = observation[0]
    vy = observation[3]
    theta = observation[4]
    contact_left = observation[6]
    contact_right = observation[7]

    legs_touching = contact_left == 1 and contact_right == 1

    on_landing_pad = abs(x) <= 0.2

    stable_velocity = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)
    stable = stable_velocity and stable_orientation
 
    if legs_touching and on_landing_pad and stable:
        return True
    return False

def objective_function(observation_history):
    obs = observation_history[-2]
    
    x = obs[0]          # Posição horizontal
    y = obs[1]          # Posição vertical
    vx = obs[2]         # Velocidade horizontal
    vy = obs[3]         # Velocidade vertical
    theta = obs[4]      # Orientação (ângulo)
    leg_l = obs[6]      # Contacto da perna esquerda
    leg_r = obs[7]      # Contacto da perna direita

    # Penalizações (queremos minimizar a distância, a velocidade de queda e a inclinação)
    fitness = - (abs(x) * 10.0) - (abs(y) * 10.0) - (abs(vy) * 50.0) - (abs(theta) * 20.0)
    
    # Recompensas
    fitness += (leg_l + leg_r) * 10.0

    # Bónus massivo por sucesso total
    success = check_successful_landing(observation_history[-1])
    if success:
        fitness += 1000.0

    return fitness, success

def simulate(genotype, render_mode = None, seed=None, env = None):
    #Simulates an episode of Lunar Lander, evaluating an individual
    env_was_none = env is None
    if env is None:
        env = gym.make("LunarLander-v3", render_mode =render_mode, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
        
    observation, info = env.reset(seed=seed)

    observation_history = [observation]
    for _ in range(STEPS):
        #Chooses an action based on the individual's genotype
        action = network(SHAPE, observation, genotype)
        observation, reward, terminated, truncated, info = env.step(action)        
        observation_history.append(observation)

        if terminated == True or truncated == True:
            break
    
    if env_was_none:    
        env.close()

    return objective_function(observation_history)

def evaluate(evaluationQueue, evaluatedQueue):
    #Evaluates individuals until it receives None
    #This function runs on multiple processes
    
    env = gym.make("LunarLander-v3", render_mode =None, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
    while True:
        ind = evaluationQueue.get()

        if ind is None:
            break
            
        ind['fitness'] = simulate(ind['genotype'], seed = None, env = env)[0]
                
        evaluatedQueue.put(ind)
    env.close()
    
def evaluate_population(population):
    #Evaluates a list of individuals using multiple processes
    for i in range(len(population)):
        evaluationQueue.put(population[i])
    new_pop = []
    for i in range(len(population)):
        ind = evaluatedQueue.get()
        new_pop.append(ind)
    return new_pop

def generate_initial_population():
    #Generates the initial population
    population = []
    for i in range(POPULATION_SIZE):
        #Each individual is a dictionary with a genotype and a fitness value
        #At this time, the fitness value is None
        #The genotype is a list of floats sampled from a uniform distribution between -1 and 1
        
        genotype = []
        for j in range(GENOTYPE_SIZE):
            genotype += [random.uniform(-1,1)]
        population.append({'genotype': genotype, 'fitness': None})
    return population

def parent_selection(population):
    # Seleção por Torneio (tamanho 3)
    tournament = random.sample(population, 3)
    winner = max(tournament, key=lambda ind: ind['fitness'])
    return copy.deepcopy(winner)

def crossover(p1, p2):
    # Crossover Uniforme
    child_genotype = []
    for i in range(GENOTYPE_SIZE):
        if random.random() < 0.5:
            child_genotype.append(p1['genotype'][i])
        else:
            child_genotype.append(p2['genotype'][i])
    return {'genotype': child_genotype, 'fitness': None}

def mutation(p):
    # Mutação Gaussiana
    for i in range(GENOTYPE_SIZE):
        if random.random() < PROB_MUTATION:
            p['genotype'][i] += random.gauss(0, STD_DEV)
            p['genotype'][i] = max(-1.0, min(1.0, p['genotype'][i]))
    return p  
    
def survival_selection(population, offspring):
    #reevaluation of the elite
    offspring.sort(key = lambda x: x['fitness'], reverse=True)
    p = evaluate_population(population[:ELITE_SIZE])
    new_population = p + offspring[ELITE_SIZE:]
    new_population.sort(key = lambda x: x['fitness'], reverse=True)
    return new_population    
        
def evolution():
    #Create evaluation processes
    evaluation_processes = []
    for i in range(NUM_PROCESSES):
        evaluation_processes.append(Process(target=evaluate, args=(evaluationQueue, evaluatedQueue)))
        evaluation_processes[-1].start()
    
    #Create initial population
    bests = []
    population = list(generate_initial_population())
    population = evaluate_population(population)
    population.sort(key = lambda x: x['fitness'], reverse=True)
    best = (population[0]['genotype']), population[0]['fitness']
    bests.append(best)
    
    #Iterate over generations
    for gen in range(NUMBER_OF_GENERATIONS):
        offspring = []
        
        #create offspring
        while len(offspring) < POPULATION_SIZE:
            if random.random() < PROB_CROSSOVER:
                p1 = parent_selection(population)
                p2 = parent_selection(population)
                ni = crossover(p1, p2)

            else:
                ni = parent_selection(population)
                
            ni = mutation(ni)
            offspring.append(ni)
            
        #Evaluate offspring
        offspring = evaluate_population(offspring)

        #Apply survival selection
        population = survival_selection(population, offspring)
        
        #Print and save the best of the current generation
        best = (population[0]['genotype']), population[0]['fitness']
        bests.append(best)
        print(f'Best of generation {gen}: {best[1]}')

    #Stop evaluation processes
    for i in range(NUM_PROCESSES):
        evaluationQueue.put(None)
    for p in evaluation_processes:
        p.join()
        
    #Return the list of bests
    return bests

def load_bests(fname):
    #Load bests from file
    bests = []
    with open(fname, 'r') as f:
        for line in f:
            fitness, shape, genotype = line.split('\t')
            bests.append(( eval(fitness),eval(shape), eval(genotype)))
    return bests

if __name__ == '__main__':

    #Pick a setting from below
    #--to evolve the controller--    
    #evolve = True
    #render_mode = None

    #--to test the evolved controller without visualisation--
    evolve = False
    render_mode = None

    #--to test the evolved controller with visualisation--
    #evolve = False
    #render_mode = 'human'
    
    
    if evolve:
        # Definição das 8 experiências da Tabela 2 
        experiencias = [
            {'id': 1, 'mut': 0.008, 'cross': 0.5, 'elite': 0},
            {'id': 2, 'mut': 0.05,  'cross': 0.5, 'elite': 0},
            {'id': 3, 'mut': 0.008, 'cross': 0.9, 'elite': 0},
            {'id': 4, 'mut': 0.05,  'cross': 0.9, 'elite': 0},
            {'id': 5, 'mut': 0.008, 'cross': 0.5, 'elite': 1},
            {'id': 6, 'mut': 0.05,  'cross': 0.5, 'elite': 1},
            {'id': 7, 'mut': 0.008, 'cross': 0.9, 'elite': 1},
            {'id': 8, 'mut': 0.05,  'cross': 0.9, 'elite': 1},
        ]
        
        n_runs = 5 # 5 repetições para significado estatístico [cite: 295, 309]
        seeds = [964, 952, 364, 913, 140] # Usamos 5 seeds fixas para consistência [cite: 307]
        
        # Iterar sobre cada experiência da Tabela 2
        for exp in experiencias:
            print(f"\n{'='*40}")
            print(f"A INICIAR EXPERIÊNCIA {exp['id']}")
            print(f"Mutação: {exp['mut']} | Crossover: {exp['cross']} | Elitismo: {exp['elite']}")
            print(f"{'='*40}\n")
            
            # Atualizar as variáveis globais
            PROB_MUTATION = exp['mut']
            PROB_CROSSOVER = exp['cross']
            ELITE_SIZE = exp['elite']
            
            # Correr as 5 repetições 
            for run in range(n_runs):
                print(f"  -> A executar repetição {run+1}/5 (Seed: {seeds[run]})")
                random.seed(seeds[run])
                bests = evolution()
                
                # Guardar os resultados com um nome de ficheiro organizado [cite: 300, 301]
                # Exemplo: log_exp1_run0.txt
                nome_ficheiro = f"log_exp{exp['id']}_run{run}.txt"
                with open(nome_ficheiro, 'w') as f:
                    for b in bests:
                        f.write(f'{b[1]}\t{SHAPE}\t{b[0]}\n')

                
    else:
        #test evolved individuals
        #pick the file to test
        filename = 'log_exp1_run0.txt'
        bests = load_bests(filename)
        b = bests[-1]
        SHAPE = b[1]
        ind = b[2]
            
        ind = {'genotype': ind, 'fitness': None}
            
            
        ntests = TEST_EPISODES

        fit, success = 0, 0
        for i in range(1,ntests+1):
            f, s = simulate(ind['genotype'], render_mode=render_mode, seed = None)
            fit += f
            success += s
        print(fit/ntests, success/ntests)
