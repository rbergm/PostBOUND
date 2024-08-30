SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postLinks AS pl,
  postHistory AS ph,
  votes AS v,
  badges AS b
WHERE p.Id = c.PostId
  AND p.Id = pl.RelatedPostId
  AND p.Id = ph.PostId
  AND p.Id = v.PostId
  AND b.UserId = c.UserId
  AND c.CreationDate >= CAST('2010-07-22 01:19:43' AS timestamp)
  AND p.Score >= -1
  AND p.ViewCount >= 0
  AND p.CommentCount <= 9
  AND ph.CreationDate >= CAST('2010-09-20 17:45:15' AS timestamp)
  AND ph.CreationDate <= CAST('2014-08-07 01:00:45' AS timestamp)
  AND v.VoteTypeId = 15;
