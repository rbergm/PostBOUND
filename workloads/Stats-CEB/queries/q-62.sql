SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postLinks AS pl,
  votes AS v,
  badges AS b,
  users AS u
WHERE p.Id = c.PostId
  AND p.Id = pl.RelatedPostId
  AND p.Id = v.PostId
  AND u.Id = p.OwnerUserId
  AND u.Id = b.UserId
  AND c.Score = 2
  AND p.ViewCount <= 7710
  AND p.CommentCount <= 12
  AND p.FavoriteCount >= 0
  AND p.FavoriteCount <= 4
  AND p.CreationDate >= CAST('2010-07-27 03:58:22' AS timestamp)
  AND u.UpVotes >= 0;
