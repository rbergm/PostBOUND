SELECT COUNT(*)
FROM posts AS p,
  postLinks AS pl,
  postHistory AS ph,
  votes AS v,
  badges AS b,
  users AS u
WHERE p.Id = pl.RelatedPostId
  AND u.Id = p.OwnerUserId
  AND u.Id = b.UserId
  AND u.Id = ph.UserId
  AND u.Id = v.UserId
  AND p.AnswerCount >= 0
  AND p.FavoriteCount >= 0
  AND pl.LinkTypeId = 1
  AND ph.PostHistoryTypeId = 2
  AND v.CreationDate >= CAST('2010-07-20 00:00:00' AS timestamp)
  AND u.Reputation >= 1
  AND u.DownVotes >= 0
  AND u.DownVotes <= 0
  AND u.UpVotes <= 439
  AND u.CreationDate <= CAST('2014-08-07 11:18:45' AS timestamp);
