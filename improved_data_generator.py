#!/usr/bin/env python3
"""
Improved Data Generator with Reduced Sparsity
Creates more realistic interaction patterns for recommendation system
"""

import csv, random, datetime
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict

# Configuration
REGIONS = ['us-east', 'us-west', 'europe', 'asia', 'oceania', 'south-america', 'middle-east', 'africa']
REGION_WEIGHTS = [0.25, 0.20, 0.15, 0.15, 0.10, 0.05, 0.05, 0.05]

WORLD_CATEGORIES = ['Social', 'Gaming', 'Education', 'Art & Creative', 'Music & Dance',
                    'Sports & Fitness', 'Adventure', 'Relaxation', 'Horror', 'Puzzle',
                    'Roleplay', 'Comedy', 'Simulation', 'Strategy', 'Sci-Fi', 'Fantasy']

LANGUAGES = ['en', 'es', 'fr', 'de', 'ja', 'ko', 'zh', 'pt', 'ru', 'it', 'ar', 'hi']
LANGUAGE_WEIGHTS = [0.4, 0.2, 0.08, 0.07, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02, 0.005, 0.005]

AGE_GROUPS = ['13-17', '18-24', '25-34', '35-44', '45-54', '55+']
AGE_WEIGHTS = [0.05, 0.35, 0.35, 0.15, 0.07, 0.03]

TAGS = ['multiplayer', 'singleplayer', 'competitive', 'casual', 'story-driven',
        'sandbox', 'educational', 'meditation', 'fitness', 'music', 'art',
        'scary', 'funny', 'realistic', 'fantasy', 'sci-fi', 'retro',
        'minimalist', 'social', 'creative', 'immersive']

# Age-based preferences for categories
AGE_CATEGORY_PREFERENCES = {
    '13-17': {'Gaming': 0.3, 'Social': 0.2, 'Adventure': 0.2, 'Music & Dance': 0.15, 'Horror': 0.15},
    '18-24': {'Gaming': 0.25, 'Social': 0.25, 'Music & Dance': 0.15, 'Adventure': 0.15, 'Horror': 0.1, 'Art & Creative': 0.1},
    '25-34': {'Social': 0.3, 'Gaming': 0.2, 'Education': 0.15, 'Art & Creative': 0.15, 'Relaxation': 0.1, 'Adventure': 0.1},
    '35-44': {'Social': 0.25, 'Education': 0.2, 'Relaxation': 0.2, 'Puzzle': 0.15, 'Strategy': 0.1, 'Gaming': 0.1},
    '45-54': {'Social': 0.25, 'Relaxation': 0.25, 'Education': 0.2, 'Puzzle': 0.15, 'Strategy': 0.15},
    '55+': {'Social': 0.3, 'Relaxation': 0.25, 'Education': 0.2, 'Puzzle': 0.15, 'Simulation': 0.1}
}

class ImprovedVRDataGenerator:
    def __init__(self):
        self.users = []
        self.worlds = []
        self.user_history = defaultdict(list)  # Track user visit history
        self.world_visitors = defaultdict(list)  # Track world visitors
        self.user_friends = defaultdict(list)  # Track user friendships
        self.world_weights = None  # Popularity weights for worlds
        self.user_weights = None  # Activity weights for users
        
    def generate_user_id(self):
        return f"U{random.randint(100000, 999999)}"
    
    def generate_world_id(self):
        return f"W{random.randint(1000000, 9999999)}"
    
    def generate_datetime(self, days_back=365):
        now = datetime.datetime.now()
        delta = datetime.timedelta(
            days=random.randint(0, days_back),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        return (now - delta).strftime('%Y-%m-%d %H:%M:%S')
    
    def generate_username(self):
        adjectives = ['Cool', 'Epic', 'Mystic', 'Shadow', 'Neon', 'Cyber', 'Dark', 'Wild', 'Hyper']
        nouns = ['Player', 'Gamer', 'Explorer', 'Creator', 'Wizard', 'Dragon', 'Ninja', 'Hero', 'Master']
        return f"{random.choice(adjectives)}{random.choice(nouns)}{random.randint(1, 9999)}"
    
    def generate_world_name(self):
        prefixes = ['The', 'Virtual', 'Meta', 'Neon', 'Quantum', 'Crystal', 'Dream', 'Galactic', 'Mystic']
        types = ['Arena', 'World', 'Realm', 'Hub', 'Oasis', 'Galaxy', 'Dungeon', 'Island', 'Kingdom']
        return f"{random.choice(prefixes)} {random.choice(types)}{random.randint(1, 9999)}"
    
    def generate_users(self, count=2000):  # Reduced from 5000
        """Generate users with preference patterns"""
        users = []
        for _ in range(count):
            age_group = random.choices(AGE_GROUPS, weights=AGE_WEIGHTS, k=1)[0]
            
            # Select preferred categories based on age group
            age_prefs = AGE_CATEGORY_PREFERENCES.get(age_group, {})
            if age_prefs:
                # Sample 2-4 preferred categories weighted by age preferences
                num_prefs = random.randint(2, 4)
                categories = list(age_prefs.keys())
                weights = list(age_prefs.values())
                preferred = random.choices(categories, weights=weights, k=num_prefs)
                preferred_categories = ','.join(list(set(preferred)))
            else:
                preferred_categories = ','.join(random.sample(WORLD_CATEGORIES, k=random.randint(2, 4)))
            
            users.append({
                'user_id': self.generate_user_id(),
                'user_name': self.generate_username(),
                'is_creator': random.choices(['true', 'false'], weights=[0.05, 0.95], k=1)[0],
                'region': random.choices(REGIONS, weights=REGION_WEIGHTS, k=1)[0],
                'preferred_language': random.choices(LANGUAGES, weights=LANGUAGE_WEIGHTS, k=1)[0],
                'age_group': age_group,
                'content_filter_level': random.choices(['none', 'low', 'medium', 'high'], 
                                                      weights=[0.4, 0.2, 0.2, 0.2], k=1)[0],
                'preferred_categories': preferred_categories
            })
        
        self.users = users
        
        # Generate user activity weights (power law distribution)
        self.user_weights = np.random.pareto(a=2.0, size=len(users))
        self.user_weights = self.user_weights / self.user_weights.sum()
        
        return users
    
    def generate_worlds(self, count=200):  # Reduced from 400
        """Generate worlds with popularity distribution"""
        if not self.users:
            raise ValueError("Users must be generated first.")
        
        creator_pool = [u for u in self.users if u['is_creator'] == 'true'] or self.users
        worlds = []
        
        for _ in range(count):
            creator = random.choice(creator_pool)
            category = random.choice(WORLD_CATEGORIES)
            
            worlds.append({
                'world_id': self.generate_world_id(),
                'world_name': self.generate_world_name(),
                'creator_name': creator['user_name'],
                'category': category,
                'sub_category': random.choice(['Featured', 'Popular', 'New', 'Classic']),
                'tags': ','.join(random.sample(TAGS, k=random.randint(2, 5))),
                'avg_session_minutes': round(random.uniform(5, 120), 1),
                'rating': round(random.uniform(3.0, 5.0), 1),
                'total_ratings': random.randint(10, 10000),
                'likes': random.randint(10, 50000),
                'max_players': random.choice([4, 8, 16, 32, 50, 100]),
                'current_players': random.randint(0, 50),
                'maturity_rating': random.choices(['E', 'E10+', 'T', 'M'], 
                                                 weights=[0.3, 0.3, 0.2, 0.2], k=1)[0],
                'is_featured': random.choice(['true', 'false', 'false'])
            })
        
        self.worlds = worlds
        
        # Generate world popularity weights (power law distribution)
        self.world_weights = np.random.pareto(a=1.5, size=len(worlds))
        self.world_weights = self.world_weights / self.world_weights.sum()
        
        return worlds
    
    def generate_user_actions_improved(self, count=150000):  # Increased from 40000
        """Generate actions with improved patterns to reduce sparsity"""
        if not self.users or not self.worlds:
            raise ValueError("Users & Worlds must be generated first.")
        
        actions = []
        action_types = ['world_join', 'world_leave', 'world_like', 'world_rate',
                       'world_create', 'world_report', 'item_purchase', 'achievement_unlock']
        action_weights = [0.4, 0.3, 0.1, 0.05, 0.01, 0.01, 0.02, 0.11]
        
        # Create user preferences mapping
        user_prefs = {}
        for user in self.users:
            preferred_cats = user['preferred_categories'].split(',')
            user_prefs[user['user_id']] = preferred_cats
        
        # Group worlds by category
        worlds_by_cat = defaultdict(list)
        for i, world in enumerate(self.worlds):
            worlds_by_cat[world['category']].append(i)
        
        # Ensure minimum interactions per user
        min_actions_per_user = 20
        users_action_count = defaultdict(int)
        
        for action_idx in range(count):
            # Select user based on activity weights
            user_idx = np.random.choice(len(self.users), p=self.user_weights)
            user = self.users[user_idx]
            user_id = user['user_id']
            
            # Determine world selection strategy
            rand = random.random()
            
            if rand < 0.3 and user_id in self.user_history and self.user_history[user_id]:
                # 30% chance: Revisit a previously visited world
                world_idx = random.choice(self.user_history[user_id])
                world = self.worlds[world_idx]
                
            elif rand < 0.5 and user_id in self.user_friends and self.user_friends[user_id]:
                # 20% chance: Visit a world that friends visited (social influence)
                friend_id = random.choice(self.user_friends[user_id])
                if friend_id in self.user_history and self.user_history[friend_id]:
                    world_idx = random.choice(self.user_history[friend_id])
                    world = self.worlds[world_idx]
                else:
                    # Fallback to popularity-based selection
                    world_idx = np.random.choice(len(self.worlds), p=self.world_weights)
                    world = self.worlds[world_idx]
                    
            elif rand < 0.8 and user_id in user_prefs:
                # 30% chance: Visit world from preferred category
                pref_cat = random.choice(user_prefs[user_id])
                if pref_cat in worlds_by_cat and worlds_by_cat[pref_cat]:
                    # Select from preferred category with popularity weighting
                    cat_indices = worlds_by_cat[pref_cat]
                    cat_weights = [self.world_weights[i] for i in cat_indices]
                    cat_weights = np.array(cat_weights) / sum(cat_weights)
                    world_idx = np.random.choice(cat_indices, p=cat_weights)
                    world = self.worlds[world_idx]
                else:
                    # Fallback to popularity-based selection
                    world_idx = np.random.choice(len(self.worlds), p=self.world_weights)
                    world = self.worlds[world_idx]
            else:
                # 20% chance: Explore based on popularity
                world_idx = np.random.choice(len(self.worlds), p=self.world_weights)
                world = self.worlds[world_idx]
            
            # Track visit history
            if user_id not in self.user_history:
                self.user_history[user_id] = []
            if world_idx not in self.user_history[user_id]:
                self.user_history[user_id].append(world_idx)
            
            # Track world visitors for potential friend connections
            self.world_visitors[world['world_id']].append(user_id)
            
            # Generate action
            action_type = random.choices(action_types, weights=action_weights, k=1)[0]
            
            if action_type == 'world_rate':
                action_value = str(random.randint(3, 5))  # Bias towards positive ratings
            elif action_type == 'world_like':
                action_value = 'true'
            elif action_type == 'world_join':
                action_value = 'joined'
            elif action_type == 'world_leave':
                action_value = 'left'
            elif action_type == 'world_report':
                action_value = random.choice(['spam', 'abuse', 'other'])
            elif action_type == 'item_purchase':
                action_value = random.choice(['cheap', 'normal', 'expensive'])
            elif action_type == 'achievement_unlock':
                action_value = random.choice(['common', 'uncommon', 'rare', 'epic', 'legendary'])
            else:
                action_value = ''
            
            actions.append({
                'action_id': f"A{random.randint(100000, 999999)}",
                'user_id': user_id,
                'world_id': world['world_id'],
                'action_type': action_type,
                'action_timestamp': self.generate_datetime(90),
                'action_value': action_value
            })
            
            users_action_count[user_id] += 1
        
        # Ensure minimum actions for users with too few
        for user in self.users:
            user_id = user['user_id']
            while users_action_count[user_id] < min_actions_per_user:
                # Add more actions for this user
                world_idx = np.random.choice(len(self.worlds), p=self.world_weights)
                world = self.worlds[world_idx]
                
                actions.append({
                    'action_id': f"A{random.randint(100000, 999999)}",
                    'user_id': user_id,
                    'world_id': world['world_id'],
                    'action_type': 'world_join',
                    'action_timestamp': self.generate_datetime(90),
                    'action_value': 'joined'
                })
                
                if world_idx not in self.user_history[user_id]:
                    self.user_history[user_id].append(world_idx)
                
                users_action_count[user_id] += 1
        
        return actions
    
    def generate_time_spent(self, count=50000):  # Increased from 20000
        """Generate time spent with correlated patterns"""
        if not self.users or not self.worlds:
            raise ValueError("Users & Worlds must be generated first.")
        
        rows = []
        seen = set()
        
        # Generate time spent for all user-world pairs that have interactions
        for user_id, world_indices in self.user_history.items():
            for world_idx in world_indices:
                world = self.worlds[world_idx]
                world_id = world['world_id']
                
                key = (user_id, world_id)
                if key in seen:
                    continue
                
                # Time spent correlates with world quality and user preferences
                base_time = world['avg_session_minutes']
                
                # Check if world is in user's preferred categories
                user = next((u for u in self.users if u['user_id'] == user_id), None)
                if user:
                    preferred_cats = user['preferred_categories'].split(',')
                    if world['category'] in preferred_cats:
                        # Spend more time in preferred categories
                        time_multiplier = random.uniform(1.2, 1.8)
                    else:
                        time_multiplier = random.uniform(0.5, 1.2)
                else:
                    time_multiplier = random.uniform(0.8, 1.2)
                
                avg_time = round(base_time * time_multiplier * random.uniform(0.8, 1.2), 1)
                avg_time = max(5.0, min(240.0, avg_time))  # Clamp between 5 and 240 minutes
                
                rows.append({
                    'user_id': user_id,
                    'world_id': world_id,
                    'avg_time_minutes': avg_time
                })
                seen.add(key)
        
        # Add some additional random time spent records if needed
        while len(rows) < count:
            user = random.choice(self.users)
            world_idx = np.random.choice(len(self.worlds), p=self.world_weights)
            world = self.worlds[world_idx]
            
            key = (user['user_id'], world['world_id'])
            if key not in seen:
                avg_time = round(random.uniform(5, 120), 1)
                rows.append({
                    'user_id': user['user_id'],
                    'world_id': world['world_id'],
                    'avg_time_minutes': avg_time
                })
                seen.add(key)
        
        return rows[:count]
    
    def generate_friend_connections(self, count=15000):  # Increased connections
        """Generate friend connections with social influence"""
        if not self.users or len(self.users) < 2:
            raise ValueError("Users must be generated first and have at least 2 users.")
        
        connections = []
        seen_pairs = set()
        
        # Create friendships based on common worlds visited
        for world_id, visitors in self.world_visitors.items():
            if len(visitors) < 2:
                continue
            
            # Users who visit the same world have a chance to become friends
            for i in range(min(20, len(visitors))):  # Limit connections per world
                if len(connections) >= count:
                    break
                
                user1 = random.choice(visitors)
                user2 = random.choice(visitors)
                
                if user1 == user2:
                    continue
                
                pair = tuple(sorted([user1, user2]))
                if pair in seen_pairs:
                    continue
                
                # Track friendship
                self.user_friends[user1].append(user2)
                self.user_friends[user2].append(user1)
                seen_pairs.add(pair)
                
                # Calculate connection strength based on common interests
                common_worlds = len(set(self.user_history.get(user1, [])) & 
                                   set(self.user_history.get(user2, [])))
                
                connections.append({
                    'connection_id': f"C{random.randint(100000, 999999)}",
                    'user_id_1': user1,
                    'user_id_2': user2,
                    'connection_date': self.generate_datetime(365),
                    'connection_strength': min(100, common_worlds * 10 + random.randint(0, 50)),
                    'mutual_friends_count': random.randint(0, 20),
                    'interactions_count': common_worlds * random.randint(5, 20),
                    'last_interaction': self.generate_datetime(7),
                    'favorite_world_together': world_id,
                    'total_time_together_hours': round(random.uniform(0, 100) * common_worlds, 1)
                })
        
        # Add remaining random connections if needed
        while len(connections) < count:
            user1 = random.choice(self.users)['user_id']
            user2 = random.choice(self.users)['user_id']
            
            if user1 == user2:
                continue
            
            pair = tuple(sorted([user1, user2]))
            if pair in seen_pairs:
                continue
            
            seen_pairs.add(pair)
            self.user_friends[user1].append(user2)
            self.user_friends[user2].append(user1)
            
            connections.append({
                'connection_id': f"C{random.randint(100000, 999999)}",
                'user_id_1': user1,
                'user_id_2': user2,
                'connection_date': self.generate_datetime(365),
                'connection_strength': random.randint(1, 50),
                'mutual_friends_count': random.randint(0, 10),
                'interactions_count': random.randint(0, 100),
                'last_interaction': self.generate_datetime(7),
                'favorite_world_together': random.choice(self.worlds)['world_id'] if self.worlds else '',
                'total_time_together_hours': round(random.uniform(0, 50), 1)
            })
        
        return connections[:count]
    
    def save_to_csv(self, filename, data):
        """Save data to CSV file"""
        if not data:
            print(f"Empty data for {filename}")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        print(f"✓ Saved {len(data)} rows to {filename}")
    
    def print_statistics(self):
        """Print statistics about the generated data"""
        print("\n" + "="*60)
        print("📊 GENERATED DATA STATISTICS")
        print("="*60)
        
        # Calculate sparsity
        total_possible = len(self.users) * len(self.worlds)
        actual_pairs = len(self.user_history)
        avg_worlds_per_user = np.mean([len(worlds) for worlds in self.user_history.values()])
        
        print(f"Users: {len(self.users)}")
        print(f"Worlds: {len(self.worlds)}")
        print(f"Total possible pairs: {total_possible:,}")
        print(f"Users with interactions: {actual_pairs:,}")
        print(f"Average worlds per user: {avg_worlds_per_user:.1f}")
        
        # Calculate expected sparsity
        unique_pairs = sum(len(worlds) for worlds in self.user_history.values())
        expected_sparsity = 1 - (unique_pairs / total_possible)
        print(f"Expected sparsity: {expected_sparsity:.1%}")
        print(f"Expected coverage: {(1-expected_sparsity):.1%}")
        
        # Friend network stats
        avg_friends = np.mean([len(friends) for friends in self.user_friends.values()]) if self.user_friends else 0
        print(f"Average friends per user: {avg_friends:.1f}")
        
        print("="*60)

def main():
    """Generate improved dataset with reduced sparsity"""
    gen = ImprovedVRDataGenerator()
    
    print("🚀 Generating improved VR dataset with reduced sparsity...")
    print("="*60)
    
    # Generate data
    users = gen.generate_users(2000)  # Reduced to 2000 users
    gen.save_to_csv('users.csv', users)
    
    worlds = gen.generate_worlds(200)  # Reduced to 200 worlds
    gen.save_to_csv('vrworld.csv', worlds)
    
    actions = gen.generate_user_actions_improved(150000)  # Increased to 150k actions
    gen.save_to_csv('user_actions.csv', actions)
    
    time_spent = gen.generate_time_spent(50000)  # Increased time records
    gen.save_to_csv('time_spent.csv', time_spent)
    
    connections = gen.generate_friend_connections(15000)  # More connections
    gen.save_to_csv('connections.csv', connections)
    
    # Print statistics
    gen.print_statistics()
    
    print("\n✅ Dataset generation complete!")
    print("The new dataset should have significantly reduced sparsity.")

if __name__ == "__main__":
    main()