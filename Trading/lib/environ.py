import gym, gym.spaces, enum
from gym.utils import seeding
import numpy as np

from . import data

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1

class Actions(enum.Enum):
	skip = 0
	Buy = 1
	Close = 2

class State:
	def __init__(self, bars_count, commission_prec, reset_on_close, reward_on_close=True, volumes=True):
		assert isinstance(bars_count,int)
		assert bars_count > 0
		assert isinstance(commission_prec, float)
		assert commission_prec >= 0.0
		assert isinstance(reset_on_close, bool)
		assert isinstance(reward_on_close, bool)
		self.bars_count = bars_count
		self.commission_prec = commission_prec
		self.reset_on_close = reset_on_close
		self.reward_on_close = reward_on_close
		self.volumes = volumes

	def reset(self, prices, offset):
		assert isinstance(prices, data.Prices)
		assert offset >=self.bars_count-1
		self.have_postion = False
		self.open_price = 0.0
		self._prices = prices
		self._offset = offset

	@property
	def shape(self):
		#[h, l, c] * bars + position_flag + rel_profit (since open) 
		if self.volumes:
			return (4 * self.bars_count + 1 + 1, )
		else:
			return (3*self.bars_count + 1 + 1, )

	def encode(self):
		res = np.ndarray(shape=self.shape, dtype=np.float32)
		shift = 0
		for bar_idx in range(-self.bars_count+1, 1):
			res[shift] = self._prices.high[self._offset + bar_idx]
			shift += 1
			res[shift] = self._prices.low[self._offset + bar_idx]
			shift += 1
			res[shift] = self._prices.close[self._offset + bar_idx]
			shift += 1
			if self.volumes:
				res[shift] = self._prices.volume[self._offset + bar_idx]
				shift += 1
		res[shift] = float(self.have_postion)
		shift += 1
		if not self.have_postion:
			res[shift] = 0.0
		else:
			res[shift] = (self._cur_close() - self.open_price) / self.open_price
		return res

	def _cur_close(self):
		open = self._prices.open[self._offset]
		rel_close = self._prices.close[self._offset]
		return open * (1.0 + rel_close)

	def step(self, action):
		assert isinstance(action, Actions)
		reward = 0.0
		done = False
		close = self._cur_close()
		if action == Actions.Buy and not self.have_postion:
			self.have_postion = True
			self.open_price = close
			reward -= self.commission_prec
		elif action == Actions.Close and self.have_postion:
			reward -= self.commission_prec
			done |= self.reset_on_close
			if self.reward_on_close:
				reward += 100.0 *(close - self.open_price) / self.open_price
			self.have_postion = False
			self.open_price = 0.0

		self._offset += 1
		prev_close = close
		close = self._cur_close()
		done |= self._offset >= self._prices.close.shape[0]-1

		if self.have_postion and not self.reward_on_close:
			reward += 100.0 * (close -prev_close) / prev_close
		return reward, done

class State1D(State):
	@property
	def shape(self):
		if self.volumes:
			return (6, self.bars_count)
		else:
			return (5, self.bars_count)

	def encode(self):
		res = np.zeros(self.shape, dtype=np.float32)
		ofs = self.bars_count-1
		res[0] = self._prices.high[self._offset-ofs:self._offset+1]
		res[1] = self._prices.low[self._offset-ofs:self._offset+1]
		res[2] = self._prices.close[self._offset-ofs:self._offset+1]
		if self.volumes:
			res[3] = self._prices.volume[self._offset-ofs:self._offset+1]
			dst = 4
		else:
			dst = 3
		if self.have_position:
			res[dst] = 1.0
			res[dst+1] = (self._cur_close() - self.open_price) / self.open_price
		return res

class StocksEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self, prices, bars_count=DEFAULT_BARS_COUNT, commission=DEFAULT_COMMISSION_PERC, reset_on_close=True, state_1d=False,random_ofs_on_reset=True, reward_on_close=False, volumes=False):
		assert isinstance(prices,dict)
		self._prices = prices
		if state_1d:
			self._state = State1D(bars_count, commission, reset_on_close, reward_on_close=reward_on_close, volumes=volumes)
		else:
			self._state = State(bars_count, commission, reset_on_close, reward_on_close=reward_on_close,volumes=volumes)

		self.action_space = gym.spaces.Discrete(n=len(Actions))
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape,dtype=np.float32)
		self.random_ofs_on_reset = random_ofs_on_reset
		self.seed()

	def reset(self):
		self._instrument = self.np_random.choice(list(self._prices.keys()))
		prices = self._prices[self._instrument]
		bars = self._state.bars_count
		if self.random_ofs_on_reset:
			offset = self.np_random.choice(prices.high.shape[0]-bars*10) + bars
		else:
			offset = bars
		self._state.reset(prices, offset)
		return self._state.encode()


	def step(self, action_idx):
		action = Actions(action_idx)
		reward, done = self._state.step(action)
		obs = self._state.encode()
		info = {"instrument":self._instrument, "offset":self._state._offset}
		return obs, reward, done, info

	def render(self,mode='human',close=False):
		pass

	def seed(self, seed=None):
		self.np_random, seed1 = seeding.np_random(seed)
		seed2 = seeding.hash_seed(seed1+1)%2 **31
		return [seed1, seed2]


	@classmethod
	def from_dir(cls, data_dir, **kwargs):
		prices = {file: data.load_relative(file) for file in data.price_files(data_dir)}
		return StocksEnv(prices, **kwargs)