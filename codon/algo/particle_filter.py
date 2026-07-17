from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List, Generator
import numpy as np


@dataclass
class Initialization:
    particles: np.ndarray  # (N, state_dim)
    weights: np.ndarray    # (N,)
    meta: dict = field(default_factory=dict)


@dataclass
class Resample:
    particles: np.ndarray  # (N, state_dim)
    weights: np.ndarray    # (N,)
    meta: dict = field(default_factory=dict)


@dataclass
class Transition:
    particles: np.ndarray  # (N, state_dim)
    meta: dict = field(default_factory=dict)


@dataclass
class Likelihood:
    weights: np.ndarray    # (N,)
    meta: dict = field(default_factory=dict)


@dataclass
class Estimate:
    mean: np.ndarray       # (state_dim,)
    mode: np.ndarray       # (state_dim,) (MAP)
    variance: np.ndarray   # (state_dim,)
    meta: dict = field(default_factory=dict)


@dataclass
class StepRecord:
    step: int
    estimate: Estimate
    particles: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    transition_meta: dict = field(default_factory=dict)
    likelihood_meta: dict = field(default_factory=dict)
    resample_meta: dict = field(default_factory=dict)


class InitializationModel(ABC):
    def __call__(self, num_particles: int, state_dim: int, *args, **kwds) -> Initialization:
        return self.forward(num_particles, state_dim, *args, **kwds)
    
    @abstractmethod
    def forward(self, num_particles: int, state_dim: int, *args, **kwds) -> Initialization:
        pass


class ResampleModel(ABC):
    def __call__(self, particles: np.ndarray, weights: np.ndarray, *args, **kwds) -> Resample:
        return self.forward(particles, weights, *args, **kwds)
    
    @abstractmethod
    def forward(self, particles: np.ndarray, weights: np.ndarray, *args, **kwds) -> Resample:
        pass


class TransitionModel(ABC):
    def __call__(self, particles: np.ndarray, *args, **kwds) -> Transition:
        return self.forward(particles, *args, **kwds)
    
    @abstractmethod
    def forward(self, particles: np.ndarray, *args, **kwds) -> Transition:
        pass


class TransitionDensityModel(TransitionModel):
    @abstractmethod
    def transition_density(self, next_particles: np.ndarray, prev_particles: np.ndarray, *args, **kwds) -> np.ndarray:
        pass


class LikelihoodModel(ABC):
    def __call__(self, particles: np.ndarray, observation: np.ndarray, *args, **kwds) -> Likelihood:
        return self.forward(particles, observation, *args, **kwds)
    
    @abstractmethod
    def forward(self, particles: np.ndarray, observation: np.ndarray, *args, **kwds) -> Likelihood:
        pass


class UniformInitialization(InitializationModel):
    def __init__(self, low: Optional[Union[float, np.ndarray]] = None, high: Optional[Union[float, np.ndarray]] = None):
        self.low = low
        self.high = high
    def forward(self, num_particles: int, state_dim: int, 
                low: Optional[Union[float, np.ndarray]] = None, 
                high: Optional[Union[float, np.ndarray]] = None, 
                *args, **kwargs) -> Initialization:
        
        l = low if low is not None else self.low
        h = high if high is not None else self.high
        
        if l is None or h is None:
            raise ValueError("UniformInitialization requires both 'low' and 'high' bounds.")
        l = np.atleast_1d(l)
        h = np.atleast_1d(h)
        assert l.shape[0] == state_dim, f'Expected low shape ({state_dim},), got {l.shape}'
        assert h.shape[0] == state_dim, f'Expected high shape ({state_dim},), got {h.shape}'
        
        particles = np.random.uniform(low=l, high=h, size=(num_particles, state_dim))
        weights = np.ones(num_particles) / num_particles
        return Initialization(
            particles=particles,
            weights=weights,
            meta={'type': 'uniform', 'low': l, 'high': h}
        )
    

class GaussianInitialization(InitializationModel):
    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[Union[float, np.ndarray]] = None):
        self.mean = mean
        self.std = std

    def forward(self, num_particles: int, state_dim: int, 
                mean: Optional[np.ndarray] = None, 
                std: Optional[Union[float, np.ndarray]] = None, 
                *args, **kwargs) -> Initialization:
        m = mean if mean is not None else self.mean
        s = std if std is not None else self.std
        
        if m is None or s is None:
            raise ValueError("GaussianInitialization requires both 'mean' and 'std'.")
        
        m = np.atleast_1d(m)
        assert m.shape[0] == state_dim, f'Expected mean shape ({state_dim},), got {m.shape}'
        
        if isinstance(s, (int, float)):
            s = np.full(state_dim, s)
        else:
            s = np.atleast_1d(s)
            
        if s.ndim == 1:
            assert s.shape[0] == state_dim, f'Expected std shape ({state_dim},), got {s.shape}'
            noise = np.random.normal(0, 1, size=(num_particles, state_dim))
            particles = m + noise * s
        elif s.ndim == 2:
            assert s.shape == (state_dim, state_dim), f'Expected cov shape ({state_dim}, {state_dim}), got {s.shape}'
            particles = np.random.multivariate_normal(mean=m, cov=s, size=num_particles)
        else:
            raise ValueError(f'std/cov dimension error: {s.ndim}D')
            
        weights = np.ones(num_particles) / num_particles
        return Initialization(particles=particles, weights=weights, meta={'type': 'gaussian', 'mean': m})
    

class ExplicitInitialization(InitializationModel):
    def __init__(self, custom_particles: Optional[np.ndarray] = None, custom_weights: Optional[np.ndarray] = None):
        self.custom_particles = custom_particles
        self.custom_weights = custom_weights

    def forward(self, num_particles: int, state_dim: int, 
                custom_particles: Optional[np.ndarray] = None, 
                custom_weights: Optional[np.ndarray] = None, 
                *args, **kwargs) -> Initialization:
        
        pts = custom_particles if custom_particles is not None else self.custom_particles
        w = custom_weights if custom_weights is not None else self.custom_weights
        if pts is None:
            raise ValueError("ExplicitInitialization requires 'custom_particles' array.")
        assert pts.shape == (num_particles, state_dim), \
            f'Expected particles shape ({num_particles}, {state_dim}), got {pts.shape}'
        if w is None:
            weights = np.ones(num_particles) / num_particles
        else:
            assert w.shape == (num_particles,), f'Expected weights shape ({num_particles},), got {w.shape}'
            weights = w / np.sum(w)
        return Initialization(
            particles=pts.copy(),
            weights=weights,
            meta={'type': 'explicit'}
        )
    

class GMMInitialization(InitializationModel):
    def __init__(self, means: List[np.ndarray], stds: List[np.ndarray], mixture_weights: Optional[List[float]] = None):
        self.means = means
        self.stds = stds
        self.mixture_weights = mixture_weights
    
    def forward(self, num_particles: int, state_dim: int, *args, **kwargs) -> Initialization:
        num_components = len(self.means)
        if self.mixture_weights is None:
            mix_w = np.ones(num_components) / num_components
        else:
            mix_w = np.array(self.mixture_weights) / np.sum(self.mixture_weights)
            
        component_counts = np.random.multinomial(num_particles, mix_w)
        
        particles_list = []
        for i, count in enumerate(component_counts):
            if count > 0:
                m = np.atleast_1d(self.means[i])
                s = np.atleast_1d(self.stds[i])
                noise = np.random.normal(0, 1, size=(count, state_dim))
                pts = m + noise * s
                particles_list.append(pts)
        particles = np.vstack(particles_list)
        weights = np.ones(num_particles) / num_particles
        return Initialization(
            particles=particles,
            weights=weights,
            meta={'type': 'gmm', 'num_components': num_components}
        )


class SystematicResampler(ResampleModel):
    def forward(self, particles: np.ndarray, weights: np.ndarray, threshold: float = 0.5, *args, **kwds) -> Resample:
        N = len(weights)
        
        neff = 1.0 / np.sum(np.square(weights))
        
        if neff >= threshold * N:
            return Resample(particles=particles.copy(), weights=weights.copy(), meta={'resampled': False, 'neff': neff})
            
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0
        
        step = 1.0 / N
        start = np.random.uniform(0, step)
        pointers = start + np.arange(N) * step
        
        indexes = np.zeros(N, dtype=int)
        i, j = 0, 0
        while i < N:
            if pointers[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
                if j >= N:
                    indexes[i:] = N - 1
                    break
                
        new_particles = particles[indexes].copy()
        new_weights = np.ones(N) / N
        return Resample(particles=new_particles, weights=new_weights, meta={'resampled': True, 'neff': neff})


class StratifiedResampler(ResampleModel):
    def forward(self, particles: np.ndarray, weights: np.ndarray, threshold: float = 0.5, *args, **kwds) -> Resample:
        N = len(weights)
        neff = 1.0 / np.sum(np.square(weights))
        
        if neff >= threshold * N:
            return Resample(particles=particles.copy(), weights=weights.copy(), meta={'resampled': False, 'neff': neff})
            
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0
        
        pointers = (np.random.random(N) + np.arange(N)) / N
        
        indexes = np.zeros(N, dtype=int)
        i, j = 0, 0
        while i < N:
            if pointers[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
                if j >= N:  # 边界防御
                    indexes[i:] = N - 1
                    break
                
        return Resample(
            particles=particles[indexes].copy(), 
            weights=np.ones(N) / N, 
            meta={'resampled': True, 'neff': neff, 'method': 'stratified'}
        )


class PFSession:
    def __init__(
        self, 
        pf: 'ParticleFilter', 
        total_steps: int, 
        keep_particles: bool = False, 
        keep_weights: bool = True
    ):
        self.pf = pf
        self.total_steps = total_steps
        self.keep_particles = keep_particles
        self.keep_weights = keep_weights
        self.history: List[StepRecord] = []
        
    def __iter__(self) -> Generator[int, None, None]:
        for step in range(self.total_steps):
            yield step
            self._record_step(step)
    
    def _record_step(self, step: int):
        est = self.pf.estimate()
        
        record = StepRecord(
            step=step,
            estimate=est,
            particles=self.pf.particles.copy() if self.keep_particles else None,
            weights=self.pf.weights.copy() if self.keep_weights else None,
            transition_meta=self.pf.last_transition_meta.copy(),
            likelihood_meta=self.pf.last_likelihood_meta.copy(),
            resample_meta=self.pf.last_resample_meta.copy()
        )
        self.history.append(record)
        
    def get_predictions(self, mode: str = 'mean') -> np.ndarray:
        if mode == 'mean':
            return np.array([r.estimate.mean for r in self.history])
        elif mode == 'mode':
            return np.array([r.estimate.mode for r in self.history])
        else:
            raise ValueError("mode must be 'mean' or 'mode'")


class ParticleFilter:
    def __init__(
        self,
        num_particles: int,
        state_dim: int,
        transition_model: TransitionModel,
        likelihood_model: LikelihoodModel,
        initialization_model: Optional[InitializationModel] = None,
        resample_model: Optional[ResampleModel] = None
    ):
        self.N = num_particles
        self.state_dim = state_dim

        self.transition_model = transition_model
        self.likelihood_model = likelihood_model
        self.initialization_model = initialization_model
        self.resample_model = resample_model if resample_model is not None else SystematicResampler()
        
        self.particles = np.zeros((self.N, self.state_dim))
        self.weights = np.ones(self.N) / self.N

        self.last_transition_meta: Dict[str, Any] = {}
        self.last_likelihood_meta: Dict[str, Any] = {}
        self.last_resample_meta: Dict[str, Any] = {}

        self.active_session: Optional[PFSession] = None
    
    def initialize(self, *args, **kwargs) -> 'ParticleFilter':
        if self.initialization_model is None:
            raise ValueError('Initialization model was not provided during PF construction.')
        
        init_data = self.initialization_model(self.N, self.state_dim, *args, **kwargs)
        
        assert init_data.particles.shape == (self.N, self.state_dim), \
            f'Expected particles shape {(self.N, self.state_dim)}, got {init_data.particles.shape}'
        assert init_data.weights.shape == (self.N,), \
            f'Expected weights shape {(self.N,)}, got {init_data.weights.shape}'
        
        self.particles = init_data.particles.copy()
        self.weights = init_data.weights / np.sum(init_data.weights)
        return self
    
    def predict(self, *args, **kwargs) -> 'ParticleFilter':
        trans_data = self.transition_model(self.particles, *args, **kwargs)
        
        assert trans_data.particles.shape == (self.N, self.state_dim), \
            f'Expected transition particles shape {(self.N, self.state_dim)}, got {trans_data.particles.shape}'
            
        self.particles = trans_data.particles
        self.last_transition_meta = trans_data.meta
        return self
    
    def update(self, observation: np.ndarray, *args, **kwargs) -> 'ParticleFilter':
        like_data = self.likelihood_model(self.particles, observation, *args, **kwargs)
        
        assert like_data.weights.shape == (self.N,), \
            f'Expected likelihood weights shape {(self.N,)}, got {like_data.weights.shape}'
            
        # P(X|Z) ∝ P(Z|X) * P(X)
        updated_weights = self.weights * (like_data.weights + 1e-30)
        self.weights = updated_weights / np.sum(updated_weights)
        
        self.last_likelihood_meta = like_data.meta
        return self

    def resample(self, *args, **kwargs) -> 'ParticleFilter':
        resample_data = self.resample_model(self.particles, self.weights, *args, **kwargs)
        self.particles = resample_data.particles
        self.weights = resample_data.weights
        self.last_resample_meta = resample_data.meta
        return self

    def estimate(self) -> Estimate:
        mean_est = np.average(self.particles, weights=self.weights, axis=0)
        
        map_idx = np.argmax(self.weights)
        mode_est = self.particles[map_idx].copy()
        
        variance_est = np.average((self.particles - mean_est)**2, weights=self.weights, axis=0)
        
        return Estimate(
            mean=mean_est,
            mode=mode_est,
            variance=variance_est,
            meta={
                'transition': self.last_transition_meta,
                'likelihood': self.last_likelihood_meta,
                'resample': self.last_resample_meta
            }
        )
    
    def forward(
        self, 
        observation: np.ndarray, 
        transition_kwargs: Optional[dict] = None, 
        likelihood_kwargs: Optional[dict] = None,
        resample_kwargs: Optional[dict] = None
    ) -> Estimate:
        trans_kw = transition_kwargs or {}
        like_kw = likelihood_kwargs or {}
        resample_kw = resample_kwargs or {}
        
        self.predict(**trans_kw)
        self.update(observation, **like_kw)
        self.resample(**resample_kw)
        
        return self.estimate()

    def new_session(
        self, 
        steps: int, 
        keep_particles: bool = False, 
        keep_weights: bool = True
    ) -> PFSession:
        self.active_session = PFSession(
            pf=self, 
            total_steps=steps, 
            keep_particles=keep_particles, 
            keep_weights=keep_weights
        )
        return self.active_session


class ParticleSmoother:
    def __init__(self, session: PFSession, transition_model: TransitionDensityModel):
        self.session = session
        self.transition_model = transition_model
        
        if not self.session.keep_particles or self.session.history[0].particles is None:
            raise ValueError("Smoothing requires 'keep_particles=True' during the PFSession.")

    def smooth(self) -> List[Estimate]:
        history = self.session.history
        T = len(history)
        N = self.session.pf.N
        state_dim = self.session.pf.state_dim

        # (T, N)
        smoothed_weights = np.zeros((T, N))
        
        # w_{T|T} = w_T
        smoothed_weights[-1, :] = history[-1].weights

        for t in reversed(range(T - 1)):
            x_curr = history[t].particles         # x_t, (N, state_dim)
            w_curr = history[t].weights           # w_t, (N,)
            
            x_next = history[t+1].particles       # x_{t+1}, (N, state_dim)
            w_smooth_next = smoothed_weights[t+1] # w_{t+1|T}, (N,)

            # P[i, j] = p(x_{t+1}^{(j)} | x_t^{(i)})，(N, N)
            P = self.transition_model.transition_density(x_next, x_curr)

            # sum_k (w_t^{(k)} * p(x_{t+1}^{(j)} | x_t^{(k)}))
            # (N,)
            denominator = w_curr @ P + 1e-30

            # w_{t+1|T}^{(j)} / denominator_j
            ratio = w_smooth_next / denominator

            # w_{t|T}^{(i)} = w_t^{(i)} * sum_j ( P[i, j] * ratio[j] )
            smoothed_weights[t, :] = w_curr * (P @ ratio)
            
            smoothed_weights[t, :] /= np.sum(smoothed_weights[t, :])

        smoothed_estimates = []
        for t in range(T):
            particles = history[t].particles
            weights = smoothed_weights[t]

            mean_est = np.average(particles, weights=weights, axis=0)
            
            map_idx = np.argmax(weights)
            mode_est = particles[map_idx].copy()
            
            variance_est = np.average((particles - mean_est)**2, weights=weights, axis=0)

            est = Estimate(
                mean=mean_est,
                mode=mode_est,
                variance=variance_est,
                meta={'smoothed': True}
            )
            smoothed_estimates.append(est)

        return smoothed_estimates