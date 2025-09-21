import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# 딥러닝 프레임워크 임포트
TENSORFLOW_AVAILABLE = False
PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow 사용 가능")
except ImportError:
    pass

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
    print(f"PyTorch 사용 가능 - CUDA: {torch.cuda.is_available()}")
except ImportError:
    pass

def load_data(filename):
    """데이터 파일을 로드하고 사용자-아이템 평점 매트릭스를 반환합니다."""
    data = []
    user_ratings = defaultdict(dict)
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            user_id = int(parts[0])
            item_id = int(parts[1])
            rating = float(parts[2])
            
            data.append((user_id, item_id, rating))
            user_ratings[user_id][item_id] = rating
    
    return data, user_ratings

def load_items(filename):
    """아이템 정보를 로드합니다 (장르 정보 포함)."""
    items = {}
    
    with open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.strip().split('|')
            item_id = int(parts[0])
            title = parts[1]
            # 장르 정보는 마지막 19개 필드 (0 또는 1)
            genres = [int(x) for x in parts[5:]]
            items[item_id] = {
                'title': title,
                'genres': np.array(genres)
            }
    
    return items

def load_user_info(filename):
    """사용자 정보를 로드합니다."""
    users = {}
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                user_id = int(parts[0])
                age = int(parts[1])
                gender = parts[2]
                occupation = parts[3]
                zip_code = parts[4]
                
                users[user_id] = {
                    'age': age,
                    'gender': gender,
                    'occupation': occupation,
                    'zip_code': zip_code
                }
    except FileNotFoundError:
        print("사용자 정보 파일을 찾을 수 없습니다.")
        return {}
    
    return users

if PYTORCH_AVAILABLE:
    class GraphConvolution(nn.Module):
        """Graph Convolution Layer"""
        def __init__(self, in_features, out_features, num_edge_types, bias=True):
            super(GraphConvolution, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.num_edge_types = num_edge_types
            
            # Edge-type specific weight matrices
            self.weight = nn.Parameter(torch.FloatTensor(num_edge_types, in_features, out_features))
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(out_features))
            else:
                self.register_parameter('bias', None)
            
            self.reset_parameters()
            
        def reset_parameters(self):
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
                
        def forward(self, features, adjacency_matrices, node_type='user'):
            """
            features: Node features (N x D)
            adjacency_matrices: List of adjacency matrices for each edge type
            """
            outputs = []
            
            for r in range(self.num_edge_types):
                if adjacency_matrices[r] is not None and adjacency_matrices[r]._nnz() > 0:
                    # Message passing for edge type r
                    support = torch.mm(features, self.weight[r])
                    # PyTorch 2.0+ compatible sparse matrix multiplication
                    output = torch.sparse.mm(adjacency_matrices[r], support)
                    outputs.append(output)
            
            if outputs:
                output = torch.stack(outputs).mean(dim=0)  # Average across edge types
            else:
                output = torch.zeros(features.size(0), self.out_features).to(features.device)
                
            if self.bias is not None:
                output = output + self.bias
                
            return output

    class GCMCEncoder(nn.Module):
        """Graph Convolutional Matrix Completion Encoder"""
        def __init__(self, num_users, num_items, num_edge_types, 
                     emb_dim, hidden_dims, dropout=0.5, use_features=False,
                     user_feature_dim=0, item_feature_dim=0):
            super(GCMCEncoder, self).__init__()
            
            self.num_users = num_users
            self.num_items = num_items
            self.num_edge_types = num_edge_types
            self.emb_dim = emb_dim
            self.use_features = use_features
            
            # Node embeddings (one-hot if no features)
            if not use_features:
                self.user_embedding = nn.Embedding(num_users, emb_dim)
                self.item_embedding = nn.Embedding(num_items, emb_dim)
                input_dim = emb_dim
            else:
                # Feature transformation layers
                self.user_feature_transform = nn.Linear(user_feature_dim, emb_dim)
                self.item_feature_transform = nn.Linear(item_feature_dim, emb_dim)
                input_dim = emb_dim
            
            # Graph convolution layers
            self.gc1 = GraphConvolution(input_dim, hidden_dims[0], num_edge_types)
            self.gc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
            
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.ReLU()
            
        def forward(self, user_ids, item_ids, adjacency_matrices, 
                    user_features=None, item_features=None):
            """
            Forward pass through encoder
            """
            # Get initial node features
            if self.use_features:
                user_emb = self.user_feature_transform(user_features)
                item_emb = self.item_feature_transform(item_features)
            else:
                user_emb = self.user_embedding(user_ids)
                item_emb = self.item_embedding(item_ids)
            
            # Concatenate user and item embeddings
            features = torch.cat([user_emb, item_emb], dim=0)
            
            # Graph convolution
            h = self.activation(self.gc1(features, adjacency_matrices))
            h = self.dropout(h)
            h = self.gc2(h)
            
            # Split back into user and item representations
            user_h = h[:self.num_users]
            item_h = h[self.num_users:]
            
            return user_h, item_h

    class BilinearDecoder(nn.Module):
        """Bilinear Decoder for rating prediction"""
        def __init__(self, hidden_dim, num_edge_types, num_basis=2):
            super(BilinearDecoder, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_edge_types = num_edge_types
            self.num_basis = num_basis
            
            # Basis weight matrices
            self.basis_matrices = nn.Parameter(
                torch.FloatTensor(num_basis, hidden_dim, hidden_dim)
            )
            
            # Coefficients for linear combination
            self.coefficients = nn.Parameter(
                torch.FloatTensor(num_edge_types, num_basis)
            )
            
            self.reset_parameters()
            
        def reset_parameters(self):
            nn.init.xavier_uniform_(self.basis_matrices)
            nn.init.xavier_uniform_(self.coefficients)
            
        def forward(self, user_emb, item_emb, edge_type=None):
            """
            Predict ratings given user and item embeddings
            """
            # Compute Q matrices for each rating type
            Q_matrices = []
            for r in range(self.num_edge_types):
                Q_r = torch.zeros(self.hidden_dim, self.hidden_dim).to(user_emb.device)
                for s in range(self.num_basis):
                    Q_r += self.coefficients[r, s] * self.basis_matrices[s]
                Q_matrices.append(Q_r)
            
            # Compute scores for all rating types
            scores = []
            for r in range(self.num_edge_types):
                score = torch.sum(user_emb * torch.matmul(item_emb, Q_matrices[r].t()), dim=1)
                scores.append(score)
            
            scores = torch.stack(scores, dim=1)  # (N, num_edge_types)
            
            # Apply softmax to get probabilities
            probs = F.softmax(scores, dim=1)
            
            # Compute expected rating
            rating_values = torch.arange(1, self.num_edge_types + 1).float().to(user_emb.device)
            predicted_ratings = torch.sum(probs * rating_values, dim=1)
            
            return predicted_ratings, probs

    class GCMC(nn.Module):
        """Graph Convolutional Matrix Completion Model"""
        def __init__(self, num_users, num_items, num_edge_types,
                     emb_dim=128, hidden_dims=[128, 64], dropout=0.5,
                     use_features=False, user_feature_dim=0, item_feature_dim=0,
                     num_basis=2):
            super(GCMC, self).__init__()
            
            self.encoder = GCMCEncoder(
                num_users, num_items, num_edge_types,
                emb_dim, hidden_dims, dropout, use_features,
                user_feature_dim, item_feature_dim
            )
            
            self.decoder = BilinearDecoder(hidden_dims[-1], num_edge_types, num_basis)
            
            # 더 나은 weight initialization
            self.apply(self._init_weights)
            
        def _init_weights(self, module):
            """Initialize weights with proper scaling"""
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Normal initialization for embeddings
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.Parameter):
                # Xavier uniform for parameter matrices
                if len(module.shape) >= 2:
                    nn.init.xavier_uniform_(module)
                else:
                    nn.init.normal_(module, mean=0, std=0.1)
            
        def forward(self, user_ids, item_ids, adjacency_matrices,
                    user_features=None, item_features=None):
            # Encode
            user_emb, item_emb = self.encoder(
                user_ids, item_ids, adjacency_matrices,
                user_features, item_features
            )
            
            # Decode - get embeddings for specific user-item pairs
            if len(user_ids.shape) == 0:  # Single prediction
                user_ids = user_ids.unsqueeze(0)
                item_ids = item_ids.unsqueeze(0)
                
            batch_user_emb = user_emb[user_ids]
            batch_item_emb = item_emb[item_ids]
            
            # Predict ratings
            ratings, probs = self.decoder(batch_user_emb, batch_item_emb)
            
            return ratings, probs

class GraphConvolutionalMatrixCompletion:
    """GC-MC: Graph Convolutional Matrix Completion for Recommendation"""
    
    def __init__(self, num_edge_types=5, emb_dim=128, hidden_dims=[128, 64],
                 learning_rate=0.001, epochs=1000, batch_size=512, dropout=0.5,
                 num_basis=2, use_cuda=True, use_features=False, patience=50,
                 lr_scheduler_patience=20, lr_scheduler_factor=0.5, min_lr=1e-6,
                 grad_clip_value=1.0):
        """
        Initialize GC-MC model
        
        Args:
            num_edge_types: Number of rating levels (e.g., 5 for 1-5 ratings)
            emb_dim: Embedding dimension
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
            dropout: Dropout rate
            num_basis: Number of basis matrices for decoder
            use_cuda: Whether to use CUDA
            use_features: Whether to use side information
            patience: Early stopping patience
            lr_scheduler_patience: Learning rate scheduler patience
            lr_scheduler_factor: Learning rate reduction factor
            min_lr: Minimum learning rate
            grad_clip_value: Gradient clipping value
        """
        self.num_edge_types = num_edge_types
        self.emb_dim = emb_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_basis = num_basis
        self.use_features = use_features
        self.patience = patience
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.min_lr = min_lr
        self.grad_clip_value = grad_clip_value
        
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = None
        self.user_list = []
        self.item_list = []
        self.adjacency_matrices = []
        self.user_features = None
        self.item_features = None
        
    def create_adjacency_matrices(self, user_ratings, num_users, num_items):
        """Create adjacency matrices for each rating type"""
        adjacency_matrices = []
        
        for r in range(1, self.num_edge_types + 1):
            # Create sparse adjacency matrix for rating r
            indices = []
            values = []
            
            for user_id, items in user_ratings.items():
                for item_id, rating in items.items():
                    if int(rating) == r:
                        # Add edges in both directions (user->item and item->user)
                        # User to item
                        indices.append([user_id - 1, num_users + item_id - 1])
                        values.append(1.0)
                        # Item to user
                        indices.append([num_users + item_id - 1, user_id - 1])
                        values.append(1.0)
            
            if indices:
                indices = torch.LongTensor(indices).t()
                values = torch.FloatTensor(values)
                shape = (num_users + num_items, num_users + num_items)
                # PyTorch 2.0+ compatible sparse tensor creation
                adj_matrix = torch.sparse_coo_tensor(indices, values, shape).to(self.device)
                
                # Normalize (symmetric normalization)
                degrees = torch.sparse.sum(adj_matrix, dim=1).to_dense()
                degrees = torch.pow(degrees + 1e-7, -0.5)  # 안정성을 위해 작은 값 추가
                degrees[torch.isinf(degrees)] = 0
                
                # D^(-1/2) * A * D^(-1/2) - 더 안정적인 방법
                row_deg = degrees[indices[0]]
                col_deg = degrees[indices[1]]
                norm_values = values * row_deg * col_deg
                
                adj_matrix = torch.sparse_coo_tensor(indices, norm_values, shape).to(self.device)
                # COO를 CSR로 변환하여 연산 효율성 향상
                adj_matrix = adj_matrix.coalesce()
            else:
                # Empty adjacency matrix
                shape = (num_users + num_items, num_users + num_items)
                adj_matrix = torch.sparse_coo_tensor(
                    torch.LongTensor([[], []]), 
                    torch.FloatTensor([]), 
                    shape
                ).to(self.device)
            
            adjacency_matrices.append(adj_matrix)
        
        return adjacency_matrices
    
    def prepare_features(self, users, items, user_info):
        """Prepare user and item features"""
        if not self.use_features:
            return None, None
        
        # User features
        user_features = []
        for user_id in sorted(users):
            if user_id in user_info:
                info = user_info[user_id]
                # Create feature vector from user info
                age_normalized = info['age'] / 100.0
                gender_encoded = 1.0 if info['gender'] == 'M' else 0.0
                # Simple encoding - can be improved
                feature = [age_normalized, gender_encoded]
            else:
                feature = [0.0, 0.0]
            user_features.append(feature)
        
        # Item features (genres)
        item_features = []
        for item_id in sorted(items.keys()):
            if item_id in items:
                feature = items[item_id]['genres'].tolist()
            else:
                feature = [0.0] * 19  # 19 genres
            item_features.append(feature)
        
        return (torch.FloatTensor(user_features).to(self.device),
                torch.FloatTensor(item_features).to(self.device))
    
    def fit(self, user_ratings, items, user_info=None):
        """Train the GC-MC model"""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GC-MC model")
        
        print("Training GC-MC model...")
        
        # Get unique users and items
        self.user_list = sorted(list(user_ratings.keys()))
        self.item_list = sorted(list(items.keys()))
        
        num_users = len(self.user_list)
        num_items = len(self.item_list)
        
        # Create adjacency matrices
        self.adjacency_matrices = self.create_adjacency_matrices(
            user_ratings, num_users, num_items
        )
        
        # Prepare features if needed
        if self.use_features and user_info:
            self.user_features, self.item_features = self.prepare_features(
                self.user_list, items, user_info
            )
            user_feature_dim = self.user_features.shape[1]
            item_feature_dim = self.item_features.shape[1]
        else:
            user_feature_dim = 0
            item_feature_dim = 0
            self.use_features = False
        
        # Create model
        self.model = GCMC(
            num_users, num_items, self.num_edge_types,
            self.emb_dim, self.hidden_dims, self.dropout,
            self.use_features, user_feature_dim, item_feature_dim,
            self.num_basis
        ).to(self.device)
        
        # Prepare training data
        train_data = []
        for user_id, items_dict in user_ratings.items():
            user_idx = self.user_list.index(user_id)
            for item_id, rating in items_dict.items():
                item_idx = self.item_list.index(item_id)
                train_data.append((user_idx, item_idx, rating))
        
        # Create data loader
        train_data = torch.LongTensor(train_data)
        train_dataset = TensorDataset(train_data[:, 0], train_data[:, 1], train_data[:, 2])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer and scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, 
                              weight_decay=1e-5)  # L2 regularization 추가
        
        # Learning rate warmup을 위한 lambda 함수
        def lr_lambda(epoch):
            warmup_epochs = 10
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0
        
        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.lr_scheduler_factor, 
            patience=self.lr_scheduler_patience, min_lr=self.min_lr
        )
        criterion = nn.MSELoss()
        
        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        all_user_ids = torch.arange(num_users).to(self.device)
        all_item_ids = torch.arange(num_items).to(self.device)
        
        self.model.train()
        print(f"시작 학습률: {self.learning_rate}")
        print(f"총 훈련 데이터: {len(train_data)}개")
        
        for epoch in range(self.epochs):
            total_loss = 0
            grad_norm_sum = 0
            num_batches = 0
            
            for batch_users, batch_items, batch_ratings in train_loader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_ratings = batch_ratings.float().to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                predicted_ratings, _ = self.model(
                    batch_users, batch_items, self.adjacency_matrices,
                    self.user_features, self.item_features
                )
                
                # Compute loss
                loss = criterion(predicted_ratings, batch_ratings)
                
                # Backward pass
                loss.backward()
                
                # Gradient monitoring
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                grad_norm_sum += grad_norm
                num_batches += 1
                
                # NaN 체크
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN or Inf loss detected at epoch {epoch+1}")
                    continue
                
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            avg_grad_norm = grad_norm_sum / num_batches
            
            # Learning rate warmup (첫 10 epoch)
            if epoch < 10:
                warmup_scheduler.step()
            else:
                # Learning rate scheduling
                scheduler.step(avg_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 25 == 0:  # 더 자주 출력
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.2e}, '
                      f'Best Loss: {best_loss:.4f}, Grad Norm: {avg_grad_norm:.4f}')
            
            # 첫 5 epoch은 더 자세히 모니터링
            if epoch < 5:
                print(f'Epoch [{epoch+1}] - Loss: {avg_loss:.4f}, LR: {current_lr:.2e}, Grad Norm: {avg_grad_norm:.4f}')
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f'Early stopping at epoch {epoch+1}. No improvement for {self.patience} epochs.')
                break
                
            # Learning rate가 너무 작아지면 중단
            if current_lr < self.min_lr:
                print(f'Learning rate {current_lr:.2e}가 minimum threshold {self.min_lr:.2e}보다 작아졌습니다. 학습을 중단합니다.')
                break
        
        print(f"GC-MC model training completed! Best loss: {best_loss:.4f}")
    
    def predict(self, user_id, item_id):
        """Predict rating for a user-item pair"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        
        # Handle unseen users/items
        if user_id not in self.user_list:
            return 3.0  # Default rating
        if item_id not in self.item_list:
            return 3.0  # Default rating
        
        user_idx = self.user_list.index(user_id)
        item_idx = self.item_list.index(item_id)
        
        with torch.no_grad():
            user_idx_tensor = torch.LongTensor([user_idx]).to(self.device)
            item_idx_tensor = torch.LongTensor([item_idx]).to(self.device)
            
            # Get all user and item ids for encoding
            all_user_ids = torch.arange(len(self.user_list)).to(self.device)
            all_item_ids = torch.arange(len(self.item_list)).to(self.device)
            
            # Predict
            predicted_rating, _ = self.model(
                user_idx_tensor, item_idx_tensor, self.adjacency_matrices,
                self.user_features, self.item_features
            )
            
            rating = predicted_rating.cpu().item()
            
            # Clip to valid range
            return max(1.0, min(5.0, rating))

def main():
    if len(sys.argv) != 3:
        print("사용법: python recommender.py <base_file> <test_file>")
        sys.exit(1)
    
    base_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    
    # dataset/ 경로를 자동으로 추가
    base_file = f"dataset/{base_file_name}"
    test_file = f"dataset/{test_file_name}"
    items_file = "dataset/u.item"
    user_file = "dataset/u.user"
    
    # 출력 파일명 생성
    output_file = base_file_name + "_prediction.txt"
    
    print(f"훈련 데이터 로딩: {base_file}")
    train_data, user_ratings = load_data(base_file)
    
    print(f"테스트 데이터 로딩: {test_file}")
    test_data, _ = load_data(test_file)
    
    print(f"아이템 정보 로딩: {items_file}")
    items = load_items(items_file)
    
    print(f"사용자 정보 로딩: {user_file}")
    user_info = load_user_info(user_file)
    
    # PyTorch가 사용 가능한 경우 GC-MC 사용
    if PYTORCH_AVAILABLE:
        print("\n=== Graph Convolutional Matrix Completion (GC-MC) 사용 ===")
        
        # GC-MC 모델 생성 및 훈련
        recommender = GraphConvolutionalMatrixCompletion(
            num_edge_types=5,  # 1-5 ratings
            emb_dim=64,  # 더 작은 임베딩 차원으로 시작
            hidden_dims=[256, 64],  # 더 안정적인 구조
            learning_rate=0.001,  # 더 작은 초기 학습률
            epochs=2000,  # 더 많은 epoch으로 설정
            batch_size=1024,  # 더 작은 배치 크기
            dropout=0.5,  # 더 적은 드롭아웃
            num_basis=2,
            use_cuda=True,
            use_features=True,  # 사용자/아이템 특징 사용
            patience=150,  # 더 큰 patience
            lr_scheduler_patience=40,  # LR scheduler patience
            lr_scheduler_factor=0.7,  # 더 보수적인 LR reduction
            min_lr=1e-7,  # 더 작은 minimum learning rate
            grad_clip_value=0.5  # 더 강한 gradient clipping
        )
        
        recommender.fit(user_ratings, items, user_info)
    else:
        print("\nPyTorch가 설치되지 않았습니다. 기본 추천 시스템을 사용합니다.")
        # 여기에 대체 모델 구현 (기존 코드의 SimpleRecommender 등)
        raise ImportError("이 구현은 PyTorch가 필요합니다.")
    
    print("\n평점 예측 시작...")
    predictions = []
    total_tests = len(test_data)
    
    for i, (user_id, item_id, actual_rating) in enumerate(test_data):
        if i % 1000 == 0:
            print(f"진행률: {i}/{total_tests} ({i/total_tests*100:.1f}%)")
        
        predicted_rating = recommender.predict(user_id, item_id)
        predictions.append((user_id, item_id, predicted_rating))
    
    print(f"\n예측 결과를 {output_file}에 저장 중...")
    
    # 결과를 파일에 저장
    with open(output_file, 'w') as f:
        for user_id, item_id, rating in predictions:
            f.write(f"{user_id}\t{item_id}\t{rating:.6f}\n")
    
    print(f"완료! 총 {len(predictions)}개의 예측 결과가 {output_file}에 저장되었습니다.")
    
    # 모델 정보 출력
    print("\n=== GC-MC 모델 정보 ===")
    print(f"사용자 수: {len(recommender.user_list)}")
    print(f"아이템 수: {len(recommender.item_list)}")
    print(f"Edge types (rating levels): {recommender.num_edge_types}")
    print(f"임베딩 차원: {recommender.emb_dim}")
    print(f"은닉층 차원: {recommender.hidden_dims}")
    print(f"기저 행렬 수: {recommender.num_basis}")
    print(f"사용자/아이템 특징 사용: {recommender.use_features}")
    print(f"Early stopping patience: {recommender.patience}")
    print(f"Learning rate scheduler patience: {recommender.lr_scheduler_patience}")
    print(f"Gradient clipping value: {recommender.grad_clip_value}")

if __name__ == "__main__":
    main()