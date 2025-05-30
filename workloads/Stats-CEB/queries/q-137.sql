SELECT COUNT(*)
FROM comments AS c,
  postLinks AS pl,
  posts AS p,
  users AS u,
  badges AS b
WHERE u.Id = b.UserId
  AND u.Id = p.OwnerUserId
  AND p.Id = c.PostId
  AND p.Id = pl.RelatedPostId
  AND c.CreationDate <= CAST('2014-09-13 20:12:15' AS timestamp)
  AND pl.LinkTypeId = 1
  AND pl.CreationDate >= CAST('2011-09-03 21:00:10' AS timestamp)
  AND pl.CreationDate <= CAST('2014-07-30 21:29:52' AS timestamp)
  AND p.Score >= 0
  AND p.Score <= 23
  AND p.AnswerCount >= 0
  AND p.AnswerCount <= 4
  AND p.CommentCount >= 0
  AND p.CommentCount <= 10
  AND p.FavoriteCount <= 9
  AND p.CreationDate >= CAST('2010-07-22 12:17:20' AS timestamp)
  AND p.CreationDate <= CAST('2014-09-12 00:27:12' AS timestamp);
