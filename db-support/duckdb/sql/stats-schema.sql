-- Manually stripped-down and optimized version of the full (MySQL-specific) schema dump of the stats database
-- Compatible with PostgreSQL, but other systems might work as well.

-- MySQL dump 10.13  Distrib 8.0.38, for Win64 (x86_64)
--
-- Host: db.relational-data.org    Database: stats
-- ------------------------------------------------------
-- Server version	8.0.31-google

--
-- Table structure for table `users`
--

CREATE TABLE users (
  Id int NOT NULL,
  Reputation int DEFAULT NULL,
  CreationDate timestamp DEFAULT NULL,
  DisplayName varchar(255) DEFAULT NULL,
  LastAccessDate timestamp DEFAULT NULL,
  WebsiteUrl varchar(255) DEFAULT NULL,
  Location varchar(255) DEFAULT NULL,
  AboutMe text,
  Views int DEFAULT NULL,
  UpVotes int DEFAULT NULL,
  DownVotes int DEFAULT NULL,
  AccountId int DEFAULT NULL,
  Age int DEFAULT NULL,
  ProfileImageUrl varchar(255) DEFAULT NULL,

  PRIMARY KEY (Id)
);

--
-- Table structure for table `badges`
--

CREATE TABLE badges (
  Id int NOT NULL,
  UserId int DEFAULT NULL,
  Name varchar(255) DEFAULT NULL,
  Date timestamp DEFAULT NULL,

  PRIMARY KEY (Id),
  CONSTRAINT badges_UserId_fkey FOREIGN KEY (UserId) REFERENCES users (Id) 
);
CREATE INDEX badges_UserId_fkey ON badges (UserId);

--
-- Table structure for table `posts`
--

CREATE TABLE posts (
  Id int NOT NULL,
  PostTypeId int DEFAULT NULL,
  AcceptedAnswerId int DEFAULT NULL,
  CreationDate timestamp DEFAULT NULL,
  Score int DEFAULT NULL,
  ViewCount int DEFAULT NULL,
  Body text,
  OwnerUserId int DEFAULT NULL,
  LasActivityDate timestamp DEFAULT NULL,
  Title varchar(255) DEFAULT NULL,
  Tags varchar(255) DEFAULT NULL,
  AnswerCount int DEFAULT NULL,
  CommentCount int DEFAULT NULL,
  FavoriteCount int DEFAULT NULL,
  LastEditorUserId int DEFAULT NULL,
  LastEditDate timestamp DEFAULT NULL,
  CommunityOwnedDate timestamp DEFAULT NULL,
  ParentId int DEFAULT NULL,
  ClosedDate timestamp DEFAULT NULL,
  OwnerDisplayName varchar(255) DEFAULT NULL,
  LastEditorDisplayName varchar(255) DEFAULT NULL,

  PRIMARY KEY (Id),
  CONSTRAINT posts_LastEditorUserId_fkey FOREIGN KEY (LastEditorUserId) REFERENCES users (Id) ,
  CONSTRAINT posts_OwnerUserId_fkey FOREIGN KEY (OwnerUserId) REFERENCES users (Id) ,
  
  -- DuckDB currently does not support self-referencing foreign keys
  -- CONSTRAINT posts_ParentId_fkey FOREIGN KEY (ParentId) REFERENCES posts (Id)
);
CREATE INDEX posts_ParentId_fkey ON posts (ParentId);
CREATE INDEX posts_OwnerUserId_fkey ON posts (OwnerUserId);
-- CREATE INDEX posts_LastEditorUserId_fkey ON posts (LastEditorUserId);

--
-- Table structure for table `tags`
--

CREATE TABLE tags (
  Id int NOT NULL,
  TagName varchar(255) DEFAULT NULL,
  Count int DEFAULT NULL,
  ExcerptPostId int DEFAULT NULL,
  WikiPostId int DEFAULT NULL,

  PRIMARY KEY (Id),
  CONSTRAINT tags_ExcerptPostId_fkey FOREIGN KEY (ExcerptPostId) REFERENCES posts (Id) 
);
CREATE INDEX tags_ExcerptPostId_fkey ON tags (ExcerptPostId);

--
-- Table structure for table `postLinks`
--

CREATE TABLE postLinks (
  Id int NOT NULL,
  CreationDate timestamp DEFAULT NULL,
  PostId int DEFAULT NULL,
  RelatedPostId int DEFAULT NULL,
  LinkTypeId int DEFAULT NULL,

  PRIMARY KEY (Id),
  CONSTRAINT postlinks_stripped_PostId_fkey FOREIGN KEY (PostId) REFERENCES posts (Id) ,
  CONSTRAINT postlinks_stripped_RelatedPostId_fkey FOREIGN KEY (RelatedPostId) REFERENCES posts (Id) 
);
CREATE INDEX postlinks_stripped_PostId_fkey ON postLinks (PostId);
CREATE INDEX postlinks_stripped_RelatedPostId_fkey ON postLinks (RelatedPostId);

--
-- Table structure for table `postHistory`
--

CREATE TABLE postHistory (
  Id int NOT NULL,
  PostHistoryTypeId int DEFAULT NULL,
  PostId int DEFAULT NULL,
  RevisionGUID varchar(255) DEFAULT NULL,
  CreationDate timestamp DEFAULT NULL,
  UserId int DEFAULT NULL,
  Text text,
  Comment text,
  UserDisplayName varchar(255) DEFAULT NULL,

  PRIMARY KEY (Id),
  CONSTRAINT postHistory_PostId_fkey FOREIGN KEY (PostId) REFERENCES posts (Id) ,
  CONSTRAINT postHistory_UserId_fkey FOREIGN KEY (UserId) REFERENCES users (Id) 
);
CREATE INDEX postHistory_UserId_fkey ON postHistory (UserId);
CREATE INDEX postHistory_PostId_fkey ON postHistory (PostId);

--
-- Table structure for table `comments`
--

CREATE TABLE comments (
  Id int NOT NULL,
  PostId int DEFAULT NULL,
  Score int DEFAULT NULL,
  Text text,
  CreationDate timestamp DEFAULT NULL,
  UserId int DEFAULT NULL,
  UserDisplayName varchar(255) DEFAULT NULL,

  PRIMARY KEY (Id),
  CONSTRAINT comment_PostId_fkey FOREIGN KEY (PostId) REFERENCES posts (Id) ,
  CONSTRAINT comment_staged_UserId_fkey FOREIGN KEY (UserId) REFERENCES users (Id) 
);
CREATE INDEX comment_staged_UserId_fkey ON comments (UserId);
CREATE INDEX comment_PostId_fkey ON comments (PostId);

--
-- Table structure for table `votes`
--

CREATE TABLE votes (
  Id int NOT NULL,
  PostId int DEFAULT NULL,
  VoteTypeId int DEFAULT NULL,
  CreationDate date DEFAULT NULL,
  UserId int DEFAULT NULL,
  BountyAmount int DEFAULT NULL,

  PRIMARY KEY (Id),
  CONSTRAINT votes_stripped_PostId_fkey FOREIGN KEY (PostId) REFERENCES posts (Id) ,
  CONSTRAINT votes_stripped_UserId_fkey FOREIGN KEY (UserId) REFERENCES users (Id) 
);
CREATE INDEX votes_stripped_UserId_fkey ON votes (UserId);
CREATE INDEX votes_stripped_PostId_fkey ON votes (PostId);

-- Dump completed on 2024-08-07 12:50:36
