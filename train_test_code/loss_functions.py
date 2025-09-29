import torch
from torch.nn import functional as F

class ContrastiveLoss(torch.nn.Module):

	def __init__(self):
		super(ContrastiveLoss, self).__init__()
		self.eps=1e-6

	def forward(self, a, b):

		pdist = torch.nn.PairwiseDistance(2)
		loss_contrastive = pdist(a,b)
		loss_contrastive = torch.mean(loss_contrastive)

		return loss_contrastive

class RMSELoss(torch.nn.Module):
	def __init__(self, eps=1e-6):
		super().__init__()
		self.mse = torch.nn.MSELoss()
		self.eps = eps
		
	def forward(self,yhat,y):
		loss = torch.sqrt(self.mse(yhat,y) + self.eps)
		return loss

class CosineDistance(torch.nn.Module):
	

	def __init__(self, dim: int = 1, eps: float = 1e-8):
		super(CosineDistance, self).__init__()
		self.dim = dim
		self.eps = eps

	def forward(self, x1, x2):
		return 1 - F.cosine_similarity(x1, x2, self.dim, self.eps)

"""
class NT_Xent(torch.nn.Module):
	def __init__(self, batch_size, temperature):
		super(NT_Xent, self).__init__()
		self.batch_size = batch_size
		self.temperature = temperature
		self.world_size = 1

		self.mask = self.mask_correlated_samples(batch_size, self.world_size)
		self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
		self.similarity_f = torch.nn.CosineSimilarity(dim=2)

	def mask_correlated_samples(self, batch_size, world_size):
		N = 2 * batch_size * world_size
		mask = torch.ones((N, N), dtype=bool)
		mask = mask.fill_diagonal_(0)
		for i in range(batch_size * world_size):
			mask[i, batch_size * world_size + i] = 0
			mask[batch_size * world_size + i, i] = 0
		return mask

	def forward(self, z_i, z_j):

		#We do not sample negative examples explicitly.
		#Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.

		N = 2 * self.batch_size * self.world_size

		z = torch.cat((z_i, z_j), dim=0)
		

		sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

		sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
		sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

		# We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
		positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
		negative_samples = sim[self.mask].reshape(N, -1)

		labels = torch.zeros(N).to(positive_samples.device).long()
		logits = torch.cat((positive_samples, negative_samples), dim=1)
		loss = self.criterion(logits, labels)
		loss /= N
		return loss
"""
"""
class NT_Xent(torch.nn.Module):
	def __init__(self, batch_size, temperature):
		super(NT_Xent, self).__init__()
		self.batch_size = batch_size
		self.temperature = temperature
		self.world_size = 1

		self.mask = self.mask_correlated_samples(batch_size, self.world_size)
		self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
		self.similarity_f = torch.nn.CosineSimilarity(dim=2)

	def mask_correlated_samples(self, batch_size, world_size):
		N = 2 * batch_size * world_size
		mask = torch.ones((N, N), dtype=bool)
		mask.fill_diagonal_(0)
		for i in range(batch_size * world_size):
			j = i + batch_size * world_size
			mask[i, j] = 0
			mask[j, i] = 0
		return mask

	def forward(self, z_i, z_j, labels=None):
		N = 2 * self.batch_size * self.world_size
		z = torch.cat((z_i, z_j), dim=0)
		device = z.device

		sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

		# Canonical SimCLR positive pairs
		sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
		sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)
		positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

		negative_samples = sim[self.mask].reshape(N, -1)

		if labels is not None:
			all_labels = torch.cat((labels, labels), dim=0)
			label_mask = (all_labels.unsqueeze(0) == all_labels.unsqueeze(1)).to(device)
			label_mask.fill_diagonal_(False)

			for i in range(self.batch_size * self.world_size):
				j = i + self.batch_size * self.world_size
				label_mask[i, j] = False
				label_mask[j, i] = False

			supervised_pos = sim[label_mask].unsqueeze(1)  # (M, 1) additional positives
			if supervised_pos.numel() > 0:
				positive_samples = torch.cat((positive_samples, supervised_pos), dim=0)

		total_pos = positive_samples.size(0)

		# Ensure negatives match positive rows
		if total_pos == N:
			final_negatives = negative_samples
		else:
			reps = (total_pos + N - 1) // N  # ceil division
			final_negatives = negative_samples.repeat(reps, 1)[:total_pos]

		logits = torch.cat((positive_samples, final_negatives), dim=1)
		targets = torch.zeros(total_pos, dtype=torch.long, device=device)

		loss = self.criterion(logits, targets) / total_pos
		return loss
"""
class Supervised_NT_Xent(torch.nn.Module):
	def __init__(self, batch_size, temperature):
		super(Supervised_NT_Xent, self).__init__()
		self.batch_size = batch_size
		self.temperature = temperature
		self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
		self.similarity_f = torch.nn.CosineSimilarity(dim=2)

	def forward(self, z_i, z_j, labels):
		N = 2 * self.batch_size
		z = torch.cat([z_i, z_j], dim=0)
		labels = labels.repeat(2)

		sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature  # (2N, 2N)

		# Remove self-similarity
		mask = torch.eye(N, dtype=torch.bool, device=z.device)
		sim.masked_fill_(mask, -float('inf'))

		# Build positive indices
		pos_idx = torch.arange(self.batch_size, device=z.device)
		pos_idx_i = pos_idx
		pos_idx_j = pos_idx + self.batch_size
		positives = torch.cat([
			sim[pos_idx_i, pos_idx_j].unsqueeze(1),
			sim[pos_idx_j, pos_idx_i].unsqueeze(1)
		], dim=0)  # (2N, 1)

		# Build negative mask (different labels only)
		label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
		negative_mask = ~label_matrix & ~mask  # (2N, 2N)

		# For each row, get only valid negatives (using masking and fill)
		logits = torch.full_like(sim, -float('inf'))  # initialize everything to -inf
		logits[negative_mask] = sim[negative_mask]    # fill in valid negatives
		logits = torch.cat([positives, logits], dim=1)  # prepend positives

		# Build labels (positive is always at index 0)
		targets = torch.zeros(N, dtype=torch.long, device=z.device)

		loss = self.criterion(logits, targets)
		return loss / N
	
"""
class NT_Xent(torch.nn.Module):
	def __init__(self, batch_size, temperature):
		super(NT_Xent, self).__init__()
		self.batch_size = batch_size
		self.temperature = temperature
		self.world_size = 1
		self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
		self.similarity_f = torch.nn.CosineSimilarity(dim=2)

	def mask_correlated_samples(self, batch_size, world_size, labels=None):
		N = 2 * batch_size * world_size
		mask = torch.ones((N, N), dtype=torch.bool)
		mask.fill_diagonal_(0)

		for i in range(batch_size * world_size):
			j = i + batch_size * world_size
			mask[i, j] = 0
			mask[j, i] = 0

		if labels is not None:
			labels = labels.view(-1, 1)
			labels = torch.cat((labels, labels), dim=0)
			labels_eq = torch.eq(labels, labels.T).to(mask.device)
			mask &= ~labels_eq

		return mask

	def forward(self, z_i, z_j, labels=None):
		device = z_i.device
		batch_size = self.batch_size
		N = 2 * batch_size * self.world_size

		z = torch.cat((z_i, z_j), dim=0)  # (2B, D)
		sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature  # (2B, 2B)
		mask = self.mask_correlated_samples(batch_size, self.world_size, labels)
		# Remove self-similarity
		

		if labels is None:
			sim = sim.masked_fill(torch.eye(N, dtype=torch.bool, device=device), float('-inf'))
			# Use default SimCLR logic with one positive per anchor
			sim_i_j = torch.diag(sim, batch_size)
			sim_j_i = torch.diag(sim, -batch_size)
			positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
			mask = self.mask_correlated_samples(batch_size, self.world_size)
			negative_samples = sim[mask].reshape(N, -1)
			logits = torch.cat((positive_samples, negative_samples), dim=1)
			targets = torch.zeros(logits.size(0), dtype=torch.long, device=device)
			loss = self.criterion(logits, targets) / N

		else:

			# Normalize logits
			logits_max, _ = torch.max(sim, dim=1, keepdim=True)
			sim = sim - logits_max.detach()  # for numerical stability
			exp_sim = torch.exp(sim)
			log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)  # log-softmax
			# Compute mean log-prob over positives
			mean_log_prob_pos = (mask.to(device) * log_prob.to(device)).sum(dim=1) / mask.sum(dim=1).clamp(min=1).to(device)
			# Loss
			loss = -mean_log_prob_pos.mean()

		return loss
"""
"""
class NT_Xent(torch.nn.Module):
	def __init__(self, batch_size, temperature):
		super(NT_Xent, self).__init__()
		self.batch_size = batch_size
		self.temperature = temperature
		self.world_size = 1
		self.criterion_wo_labels = torch.nn.CrossEntropyLoss(reduction="sum")
		#self.criterion_w_labels = torch.nn.BCELossWithLogits(reduction='none')

		self.similarity_f = torch.nn.CosineSimilarity(dim=2)
		
	def mask_correlated_samples(self, device, labels=None):
		
		N = 2 * self.batch_size * self.world_size
		ground_truth_labels = torch.zeros((N, N), dtype=torch.bool)
		ground_truth_labels.fill_diagonal_(0)

		for i in range(self.batch_size * self.world_size):
			j = i + self.batch_size * self.world_size
			ground_truth_labels[i, j] = 1
			ground_truth_labels[j, i] = 1

		if (labels is None):

			target = torch.argmax(ground_truth_labels.int(), dim=1).to(device)

		else:

			ground_truth_labels.fill_diagonal_(1)

			labels = labels.view(-1, 1)
			labels = torch.cat((labels, labels), dim=0)

			labels_eq = torch.eq(labels, labels.T)

			ground_truth_labels = ground_truth_labels.to(device) | labels_eq

			target = ground_truth_labels.float()

		return target


	def forward(self, z_i, z_j, labels=None):
		inf_val = 100000000
		device = z_i.device
		N = 2 * self.batch_size * self.world_size
		z = torch.cat((z_i, z_j), dim=0)  # (2B, D)
		eye = torch.eye(N)
		eye, ~eye.bool()

		xcs = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))
		
		X = xcs.clone()
		
		target = self.mask_correlated_samples(device, labels)
		
		if (labels is None):
			#X[eye.bool()] = float("-inf")
			X[eye.bool()] = float(-inf_val)
			loss = self.criterion_wo_labels(X.to(device), target)

		else:        

			X[eye.bool()] = float(inf_val)

			loss = F.binary_cross_entropy_with_logits(X.to(device) / self.temperature, target, reduction="none")
			#loss = F.binary_cross_entropy(((X.to(device) / self.temperature).sigmoid()), target, reduction="none")

			target_pos = target.bool()
			target_neg = ~target_pos


			loss_pos = torch.zeros(N, N).to(device).masked_scatter(target_pos, loss[target_pos])
			loss_neg = torch.zeros(N, N).to(device).masked_scatter(target_neg, loss[target_neg])

			loss_pos = loss_pos.sum(dim=1)
			loss_neg = loss_neg.sum(dim=1)

			num_pos = target.sum(dim=1)
			num_neg = target.size(0) - num_pos

			#print(num_pos)
			#print(num_neg)

			loss = ((loss_pos / num_pos) + (loss_neg / num_neg)).mean()
			#loss = (loss_pos + loss_neg).mean()

		return loss
"""

class NT_Xent(torch.nn.Module):
	def __init__(self, batch_size, temperature):
		super(NT_Xent, self).__init__()
		self.batch_size = batch_size
		self.temperature = temperature
		self.world_size = 1

		self.mask = self.mask_correlated_samples(batch_size, self.world_size)
		self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
		self.similarity_f = torch.nn.CosineSimilarity(dim=2)

	def mask_correlated_samples(self, batch_size, world_size):
		N = 2 * batch_size * world_size
		mask = torch.ones((N, N), dtype=bool)
		mask = mask.fill_diagonal_(0)
		for i in range(batch_size * world_size):
			mask[i, batch_size * world_size + i] = 0
			mask[batch_size * world_size + i, i] = 0
		return mask

	def forward(self, z_i, z_j):
		"""
		We do not sample negative examples explicitly.
		Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
		"""
		N = 2 * self.batch_size * self.world_size

		z = torch.cat((z_i, z_j), dim=0)

		sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

		sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
		sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

		# We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
		positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
		negative_samples = sim[self.mask].reshape(N, -1)

		labels = torch.zeros(N).to(positive_samples.device).long()
		logits = torch.cat((positive_samples, negative_samples), dim=1)
		loss = self.criterion(logits, labels)
		loss /= N
		return loss
"""
class NT_Xent_supervised(torch.nn.Module):
	def __init__(self, batch_size, temperature):
		super(NT_Xent_supervised, self).__init__()
		self.batch_size = batch_size
		self.temperature = temperature
		self.world_size = 1
		self.criterion_wo_labels = torch.nn.CrossEntropyLoss(reduction="sum")
		#self.criterion_w_labels = torch.nn.BCELossWithLogits(reduction='none')

		self.similarity_f = torch.nn.CosineSimilarity(dim=2)
		
	def mask_correlated_samples(self, device, labels=None):
		
		N = 2 * self.batch_size * self.world_size
		ground_truth_labels = torch.zeros((N, N), dtype=torch.bool)
		ground_truth_labels.fill_diagonal_(0)

		for i in range(self.batch_size * self.world_size):
			j = i + self.batch_size * self.world_size
			ground_truth_labels[i, j] = 1
			ground_truth_labels[j, i] = 1

		if (labels is None):

			target = torch.argmax(ground_truth_labels.int(), dim=1).to(device)

		else:

			ground_truth_labels.fill_diagonal_(1)

			labels = labels.view(-1, 1)
			labels = torch.cat((labels, labels), dim=0)

			labels_eq = torch.eq(labels, labels.T)

			ground_truth_labels = ground_truth_labels.to(device) | labels_eq

			target = ground_truth_labels.float()

		return target


	def forward(self, z_i, z_j, labels=None):
		inf_val = 100000000
		device = z_i.device
		N = 2 * self.batch_size * self.world_size
		z = torch.cat((z_i, z_j), dim=0)  # (2B, D)
		eye = torch.eye(N)
		eye, ~eye.bool()

		xcs = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))
		
		X = xcs.clone()
		
		target = self.mask_correlated_samples(device, labels)
		
		if (labels is None):
			#X[eye.bool()] = float("-inf")
			X[eye.bool()] = float(-inf_val)
			loss = self.criterion_wo_labels(X.to(device), target)

		else:        

			X[eye.bool()] = float(inf_val)

			loss = F.binary_cross_entropy_with_logits(X.to(device) / self.temperature, target, reduction="none")
			#loss = F.binary_cross_entropy(((X.to(device) / self.temperature).sigmoid()), target, reduction="none")

			target_pos = target.bool()
			target_neg = ~target_pos


			loss_pos = torch.zeros(N, N).to(device).masked_scatter(target_pos, loss[target_pos])
			loss_neg = torch.zeros(N, N).to(device).masked_scatter(target_neg, loss[target_neg])

			loss_pos = loss_pos.sum(dim=1)
			loss_neg = loss_neg.sum(dim=1)

			num_pos = target.sum(dim=1)
			num_neg = target.size(0) - num_pos

			#print(num_pos)
			#print(num_neg)

			loss = ((loss_pos / num_pos) + (loss_neg / num_neg)).mean()
			#loss = (loss_pos + loss_neg).mean()

		return loss
"""

class NT_Xent_supervised(torch.nn.Module):
	def __init__(self, batch_size, temperature):
		super(NT_Xent_supervised, self).__init__()
		self.batch_size = batch_size
		self.temperature = temperature
		self.world_size = 1
		self.criterion_wo_labels = torch.nn.CrossEntropyLoss(reduction="sum")
		self.criterion_w_labels = torch.nn.BCELoss(reduction='none')

		self.similarity_f = torch.nn.CosineSimilarity(dim=2)

	def mask_correlated_samples(self, device, labels=None):
		N = 2 * self.batch_size * self.world_size
		ground_truth_labels = torch.zeros((N, N), dtype=torch.bool)
		ground_truth_labels.fill_diagonal_(0)

		for i in range(self.batch_size * self.world_size):
			j = i + self.batch_size * self.world_size
			ground_truth_labels[i, j] = 1
			ground_truth_labels[j, i] = 1

		ground_truth_labels.fill_diagonal_(1)
		positive_mask = ground_truth_labels

		labels = labels.view(-1, 1)
		labels = torch.cat((labels, labels), dim=0)

		negative_mask = ~torch.eq(labels, labels.T)
		target_positive = positive_mask.float().to(device)
		target_negative = negative_mask.float().to(device)

		return target_positive, target_negative


	def forward(self, z_i, z_j, labels=None):
		inf_val = 100000000
		device = z_i.device
		N = 2 * self.batch_size * self.world_size
		z = torch.cat((z_i, z_j), dim=0)  # (2B, D)
		eye = torch.eye(N)
		#eye, ~eye.bool()

		xcs = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))
		
		X = xcs.clone()
		
		target_positive, target_negative = self.mask_correlated_samples(device, labels)
		
		X[eye.bool()] = float(inf_val)

		loss = F.binary_cross_entropy_with_logits(X.to(device) / self.temperature, target_positive, reduction="none")

		target_pos = target_positive.bool()
		target_neg = target_negative.bool()

		
		loss_pos = torch.zeros(N, N).to(device).masked_scatter(target_pos, loss[target_pos])
		loss_neg = torch.zeros(N, N).to(device).masked_scatter(target_neg, loss[target_neg])

		loss_pos = loss_pos.sum(dim=1)
		loss_neg = loss_neg.sum(dim=1)

		num_pos = target_pos.sum(dim=1)
		num_neg = target_neg.sum(dim=1)

		loss = ((loss_pos / num_pos) + (loss_neg / num_neg)).mean()
		#loss = (loss_pos + loss_neg).mean()

		return loss

def supervised_contrastive(z_img, z_txt, labels, temperature=0.07):
	# z_img and z_txt shape: (B, D)
	z = torch.cat([z_img, z_txt], dim=0)
	z = F.normalize(z, dim=1)
	sim = torch.matmul(z, z.T) / temperature
	N = z.size(0)
	mask = torch.eye(N, dtype=torch.bool, device=z.device)
	sim = sim.masked_fill(mask, float('-inf'))

	labels = labels.view(-1)
	labels = torch.cat([labels, labels], dim=0)
	pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask

	log_prob = F.log_softmax(sim, dim=1)
	loss = -(log_prob * pos_mask.float()).sum(1) / pos_mask.sum(1).clamp(min=1)
	return loss.mean()


class InfoNCE(torch.nn.Module):
	def __init__(self, temperature=0.5):
		super(InfoNCE, self).__init__()
		self.temperature = temperature

	def forward(self, z_i, z_j):
		"""
		Args:
			z_i (Tensor): Features from view 1, shape (B, D)
			z_j (Tensor): Features from view 2, shape (B, D)
		Returns:
			Scalar loss
		"""
		batch_size = z_i.size(0)
		z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
		z = F.normalize(z, dim=1)

		sim_matrix = torch.matmul(z, z.T) / self.temperature

		# Mask self-similarities
		mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
		sim_matrix = sim_matrix.masked_fill(mask, -float("inf"))

		labels = torch.arange(batch_size, device=z.device)
		labels = torch.cat([labels + batch_size, labels], dim=0)

		loss = F.cross_entropy(sim_matrix, labels)
		return loss

"""
class InfoNCE_supervised(torch.nn.Module):
	def __init__(self, temperature=0.07):
		super().__init__()
		self.temperature = temperature

	def forward(self, z_i, z_j, target=None):
		
		z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
		z = F.normalize(z, dim=1)

		sim = torch.matmul(z, z.T) / self.temperature

		batch_size = z_i.size(0)
		target = target.view(-1)
		target = torch.cat([target, target], dim=0)  # (2B,)
		mask = (target.unsqueeze(0) == target.unsqueeze(1)).float()
		mask.fill_diagonal_(0)
		

		pos_mask = mask
		neg_mask = 1 - pos_mask

		# Prevent logsumexp over irrelevant entries
		pos_mask_add = neg_mask * -1000
		neg_mask_add = pos_mask * -1000

		log_infonce_per_example = (sim * pos_mask + pos_mask_add).logsumexp(-1) - (sim * neg_mask + neg_mask_add).logsumexp(-1)
		return -log_infonce_per_example.mean()
"""

class InfoNCE_supervised(torch.nn.Module):
	def __init__(self, temperature=0.07):
		super().__init__()
		self.temperature = temperature

	def forward(self, z_i, z_j, target):
		"""
		Args:
			z_i, z_j: feature tensors of shape (B, D)
				z_i: image embeddings (student; grads)
				z_j: text embeddings (teacher; frozen)
			target: labels of shape (B,)
		Returns:
			scalar loss (mean over batch)
		"""
		batch_size = z_i.size(0)

		# Normalize embeddings
		z_i = F.normalize(z_i, dim=1)
		z_j = F.normalize(z_j, dim=1)  # freeze text encoder

		# Concatenate embeddings along batch dimension
		z = torch.cat([z_i, z_j], dim=0)  # (2B, D)

		# Compute similarity matrix scaled by temperature
		sim = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)

		# Duplicate targets for both halves
		target = target.view(-1)
		target = torch.cat([target, target], dim=0)  # (2B,)

		# Create boolean mask of positives: same class & not self
		pos_mask = (target.unsqueeze(0) == target.unsqueeze(1))
		diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=pos_mask.device)
		pos_mask = pos_mask & (~diag_mask)  # exclude self

		# Mask out self similarity by setting to -inf (safe for half precision)
		sim = sim.masked_fill(diag_mask, float("-inf"))

		# Compute stable log-sum-exp over all similarities (denominator)
		logsumexp_all = torch.logsumexp(sim, dim=1)  # (2B,)

		# Compute log-sum-exp over positive pairs only (numerator)
		sim_pos = sim.masked_fill(~pos_mask, float("-inf"))
		logsumexp_pos = torch.logsumexp(sim_pos, dim=1)  # (2B,)

		# Handle rows with no positives: replace -inf with 0 (no loss contribution)
		valid_mask = torch.isfinite(logsumexp_pos)
		losses = torch.zeros_like(logsumexp_pos)
		losses[valid_mask] = -(logsumexp_pos[valid_mask] - logsumexp_all[valid_mask])

		return losses.mean()

"""
class CLIP_Loss(torch.nn.Module):
	def __init__(self, batch_size = 4, temperature = 0.07):
		super(CLIP_Loss, self).__init__()
		
		self.batch_size = batch_size
		self.temperature = temperature
		
	def forward(self,image_embeddings,text_embeddings):

		similarity_matrix = torch.nn.functional.cosine_similarity(text_embeddings, image_embeddings.unsqueeze(0), dim=-1)
	
		diagonal = similarity_matrix.diag().view(-1, 1)
		positive_pairs = diagonal.expand_as(similarity_matrix)
		loss = 0.5 * (1 - positive_pairs) + 0.5 * torch.clamp(similarity_matrix - 0.1, min=0.0)
		loss = loss.sum() / similarity_matrix.size(0)  # Normalize by the batch size
		return loss
"""

class CLIP_Loss(torch.nn.Module):
	def __init__(self, batch_size = 4, temperature=0.07):
		super(CLIP_Loss, self).__init__()
		self.temperature = temperature

	def forward(self, image_embeddings, text_embeddings):
		# Normalize features
		#image_embeddings = torch.nn.functional.normalize(image_embeddings, dim=-1)
		#text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)

		# Similarity logits
		logits_per_image = image_embeddings @ text_embeddings.T / self.temperature
		logits_per_text = text_embeddings @ image_embeddings.T / self.temperature

		# Targets
		targets = torch.arange(image_embeddings.size(0), device=image_embeddings.device)

		# Symmetric cross-entropy loss
		loss_i2t = torch.nn.functional.cross_entropy(logits_per_image, targets)
		loss_t2i = torch.nn.functional.cross_entropy(logits_per_text, targets)

		return (loss_i2t + loss_t2i) / 2

class MultiPositiveInfoNCELoss(torch.nn.Module):
	def __init__(self, temperature=0.07):
		"""
		Multi-positive InfoNCE loss for image-to-concept alignment.

		Args:
			temperature (float): Scaling factor for similarities.
		"""
		super().__init__()
		self.temperature = temperature

	def forward(self, img_emb, key_emb, labels):
		"""
		Args:
			img_emb (Tensor): (B, D) batch of image embeddings
			key_emb (Tensor): (C, D) concept embeddings
			labels (Tensor): (B, C) binary multi-hot matrix

		Returns:
			loss (Tensor): scalar
		"""
		valid_mask = (labels.sum(dim=1) > 0)
		img_emb = img_emb[valid_mask]
		labels = labels[valid_mask]

		B, C = labels.shape

		#key_emb = key_emb.detach()

		#print(img_emb.shape, key_emb.shape, labels.shape)
		# Normalize embeddings
		img_emb = F.normalize(img_emb, dim=1)  # (B, D)
		key_emb = F.normalize(key_emb, dim=1)  # (C, D)

		# Cosine similarity
		sim = img_emb @ key_emb.T              # (B, C)
		sim = sim / self.temperature

		loss = F.binary_cross_entropy_with_logits(sim, labels.float())

		return loss


if __name__ == "__main__":
	pass